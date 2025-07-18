# -*- coding: utf-8 -*-
"""
模型训练器模块

负责模型的训练和微调。
"""

import torch
from pathlib import Path
from typing import Optional
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType

from ..config import Config, TrainingConfig
from ..data import DataLoader
from ..utils.logger import get_logger
from .downloader import ModelDownloader
from .adapters import ModelAdapterFactory


class SecurityModelTrainer:
    """网络安全模型训练器"""
    
    def __init__(self, config: Config = None, training_config: TrainingConfig = None, model_key: str = None):
        self.config = config or Config()
        self.training_config = training_config or TrainingConfig()
        self.logger = get_logger(self.__class__.__name__)
        
        # 设置当前模型
        if model_key:
            self.config.set_current_model(model_key)
        
        # 获取模型配置和适配器
        self.model_config = self.config.get_current_model_config()
        self.adapter = ModelAdapterFactory.create_adapter(self.model_config)
        
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # 初始化组件
        self.model_downloader = ModelDownloader(self.config)
        self.data_loader = DataLoader(self.config)
    
    def setup_model(self, model_name: Optional[str] = None) -> None:
        """设置模型和分词器"""
        if model_name is None:
            model_name = self.config.get_current_model_key()
        
        # 确保模型已下载
        model_path = self.model_downloader.download_model(model_name)

        # 检查模型路径是否存在且不为空
        if not model_path.exists() or not any(model_path.iterdir()):
            self.logger.error(f"模型目录不存在或为空: {model_path}")
            raise FileNotFoundError(f"模型目录不存在或为空: {model_path}")
        
        self.logger.info(f"加载模型: {model_path}")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            padding_side="right"
        )
        
        # 使用适配器设置分词器
        self.tokenizer = self.adapter.setup_tokenizer(self.tokenizer)
        
        # 配置8位量化以节省显存
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
        
        # 加载模型
        if self.model_config.quantization_compatible:
            self.model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                trust_remote_code=True,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16 if self.training_config.fp16 else torch.float32
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.float16 if self.training_config.fp16 else torch.float32
            )
        
        # 使用适配器设置模型
        self.model = self.adapter.setup_model(self.model)
        
        self.logger.info("模型已加载并配置8位量化")
        
        # 应用LoRA
        if self.training_config.use_lora and self.model_config.lora_compatible:
            self._apply_lora()
        
        self.logger.info("模型设置完成")
    
    def _apply_lora(self) -> None:
        """应用LoRA微调"""
        # 使用模型特定的目标模块
        lora_config_dict = self.training_config.lora_config.to_peft_config_dict()
        if self.model_config.lora_target_modules:
            lora_config_dict['target_modules'] = self.model_config.lora_target_modules
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            **lora_config_dict
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # 确保模型处于训练模式
        self.model.train()
        
        # 启用梯度检查点（如果配置了）
        if self.training_config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        self.model.print_trainable_parameters()
        
        self.logger.info("LoRA配置已应用")
    
    def prepare_training(self, data_file: Optional[str] = None) -> None:
        """准备训练"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("请先调用 setup_model() 设置模型")
        
        # 准备数据集
        self.train_dataset, self.eval_dataset = self.data_loader.prepare_datasets(
            self.tokenizer, data_file
        )
        
        # 设置训练参数
        training_args = TrainingArguments(
            output_dir=str(self.config.CHECKPOINTS_DIR),
            **self.training_config.to_training_arguments_dict()
        )
        
        # 数据整理器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # 创建训练器
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        self.logger.info("训练准备完成")
    
    def train(self) -> None:
        """开始训练"""
        if self.trainer is None:
            raise ValueError("请先调用 prepare_training() 准备训练")
        
        self.logger.info("开始模型训练...")
        
        # 开始训练
        self.trainer.train()
        
        self.logger.info("模型训练完成")
    
    def save_model(self, output_dir: Optional[str] = None) -> Path:
        """保存模型"""
        if self.model is None:
            raise ValueError("没有可保存的模型")
        
        if output_dir is None:
            output_dir = self.config.OUTPUT_DIR / "final_model"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(str(output_dir))
        
        # 保存分词器
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(str(output_dir))
        
        self.logger.info(f"模型已保存到: {output_dir}")
        return output_dir
    
    def evaluate(self) -> dict:
        """评估模型"""
        if self.trainer is None:
            raise ValueError("请先调用 prepare_training() 准备训练")
        
        self.logger.info("开始模型评估...")
        
        eval_results = self.trainer.evaluate()
        
        self.logger.info(f"评估结果: {eval_results}")
        return eval_results
    
    def test_model(self, test_prompts: Optional[list] = None) -> list:
        """测试模型"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("请先设置模型")
        
        if test_prompts is None:
            test_prompts = [
                "如何进行网络安全渗透测试？",
                "编写一个简单的端口扫描脚本",
                "解释SQL注入攻击的原理"
            ]
        
        results = []
        
        for prompt in test_prompts:
            self.logger.info(f"测试提示: {prompt}")
            
            # 编码输入
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            # 移动到设备
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # 生成回复
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # 解码输出
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            result = {
                "prompt": prompt,
                "response": response.strip()
            }
            
            results.append(result)
            self.logger.info(f"回复: {response.strip()}")
        
        return results
    
    def get_training_info(self) -> dict:
        """获取训练信息"""
        info = {
            "model_loaded": self.model is not None,
            "tokenizer_loaded": self.tokenizer is not None,
            "trainer_ready": self.trainer is not None,
            "config": {
                "base_model": self.config.BASE_MODEL_NAME,
                "use_lora": self.training_config.use_lora,
                "fp16": self.training_config.fp16,
                "batch_size": self.training_config.batch_size,
                "learning_rate": self.training_config.learning_rate,
                "num_epochs": self.training_config.num_epochs
            }
        }
        
        if hasattr(self, 'train_dataset'):
            info["train_dataset_size"] = len(self.train_dataset)
        
        if hasattr(self, 'eval_dataset'):
            info["eval_dataset_size"] = len(self.eval_dataset)
        
        return info