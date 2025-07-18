# -*- coding: utf-8 -*-
"""
模型推理模块

负责模型的推理和对话。
"""

import torch
from pathlib import Path
from typing import Optional, List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from ..config import Config
from ..config.model_configs import ModelRegistry
from ..utils.logger import get_logger
from .adapters import ModelAdapterFactory


class SecurityModelInference:
    """网络安全模型推理器"""
    
    def __init__(self, config: Config = None, model_key: Optional[str] = None):
        self.config = config or Config()
        self.logger = get_logger(self.__class__.__name__)
        
        # 设置模型
        if model_key:
            self.config.set_current_model(model_key)
        
        self.model_config = self.config.get_current_model_config()
        self.adapter = ModelAdapterFactory.create_adapter(self.model_config)
        
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.logger.info(f"初始化推理器，使用模型: {self.model_config.name}")
    
    def load_model(self, model_path: Optional[str] = None, 
                  base_model_path: Optional[str] = None,
                  model_key: Optional[str] = None) -> None:
        """加载模型"""
        # 如果指定了新的模型键，更新配置
        if model_key:
            self.config.set_current_model(model_key)
            self.model_config = self.config.get_current_model_config()
            self.adapter = ModelAdapterFactory.create_adapter(self.model_config)
            self.logger.info(f"切换到模型: {self.model_config.name}")
        
        if model_path is None:
            # 优先使用训练后的模型
            trained_model_path = self.config.OUTPUT_DIR / "final_model"
            if trained_model_path.exists():
                model_path = trained_model_path
            else:
                # 使用基础模型
                model_path = self.config.get_model_path()
        
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"模型路径不存在: {model_path}")
        
        self.logger.info(f"加载模型: {model_path} ({self.model_config.name})")
        
        # 检查是否是LoRA模型
        is_lora_model = (model_path / "adapter_config.json").exists()
        
        if is_lora_model:
            self._load_lora_model(model_path, base_model_path)
        else:
            self._load_full_model(model_path)
        
        self.logger.info("模型加载完成")
    
    def _load_full_model(self, model_path: Path) -> None:
        """加载完整模型"""
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True
        )
        
        # 使用适配器设置分词器
        self.tokenizer = self.adapter.setup_tokenizer(self.tokenizer)
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto" if self.device == "cuda" else None
        )
        
        # 使用适配器设置模型
        self.model = self.adapter.setup_model(self.model)
    
    def _load_lora_model(self, lora_path: Path, base_model_path: Optional[str] = None) -> None:
        """加载LoRA模型"""
        if base_model_path is None:
            # 使用当前模型配置的路径
            base_model_path = self.config.get_model_path()
        
        base_model_path = Path(base_model_path)
        
        if not base_model_path.exists():
            raise FileNotFoundError(f"基础模型路径不存在: {base_model_path}")
        
        # 加载基础模型
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(base_model_path),
            trust_remote_code=True
        )
        
        # 使用适配器设置分词器
        self.tokenizer = self.adapter.setup_tokenizer(self.tokenizer)
        
        base_model = AutoModelForCausalLM.from_pretrained(
            str(base_model_path),
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto" if self.device == "cuda" else None
        )
        
        # 使用适配器设置基础模型
        base_model = self.adapter.setup_model(base_model)
        
        # 加载LoRA适配器
        self.model = PeftModel.from_pretrained(base_model, str(lora_path))
    
    def generate_response(self, prompt: str, max_length: int = 512, 
                         temperature: float = 0.7, top_p: float = 0.9) -> str:
        """生成回复"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("请先加载模型")
        
        # 编码输入
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # 移动到设备
        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # 获取模型配置的生成参数
        gen_config = self.adapter.get_generation_config()
        
        # 合并参数，优先使用传入的参数
        generation_kwargs = {
            'max_new_tokens': max_length,
            'temperature': temperature,
            'top_p': top_p,
            'do_sample': True,
            'pad_token_id': self.tokenizer.eos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            **gen_config  # 模型特定的配置
        }
        
        # 覆盖传入的参数
        generation_kwargs.update({
            'max_new_tokens': max_length,
            'temperature': temperature,
            'top_p': top_p
        })
        
        # 生成回复
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **generation_kwargs
            )
        
        # 解码输出
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def chat(self, message: str, history: Optional[List[Dict[str, str]]] = None, 
             system_message: Optional[str] = None) -> tuple[str, List[Dict[str, str]]]:
        """对话接口"""
        if history is None:
            history = []
        
        # 构建消息列表
        messages = []
        
        # 添加系统消息
        if system_message:
            messages.append({"role": "system", "content": system_message})
        else:
            # 使用模型配置的默认身份
            messages.append({
                "role": "system", 
                "content": self.adapter.get_default_system_message()
            })
        
        # 添加历史对话
        for turn in history:
            messages.append({"role": "user", "content": turn['user']})
            messages.append({"role": "assistant", "content": turn['assistant']})
        
        # 添加当前消息
        messages.append({"role": "user", "content": message})
        
        # 使用适配器格式化对话
        adapter_prompt = self.adapter.format_chat_prompt(messages)
        
        if adapter_prompt is not None:
            # 使用适配器的自定义格式
            prompt = adapter_prompt
        elif hasattr(self.tokenizer, 'apply_chat_template') and self.model_config.supports_chat_template:
            # 使用tokenizer的chat template
            prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            # 降级到简单文本拼接
            prompt = ""
            for msg in messages:
                if msg["role"] == "system":
                    prompt += f"系统: {msg['content']}\n"
                elif msg["role"] == "user":
                    prompt += f"用户: {msg['content']}\n"
                elif msg["role"] == "assistant":
                    prompt += f"助手: {msg['content']}\n"
            prompt += "助手: "
        
        # 使用模型配置的生成参数
        gen_config = self.adapter.get_generation_config()
        response = self.generate_response(
            prompt, 
            max_length=gen_config.get('max_new_tokens', 512),
            temperature=gen_config.get('temperature', 0.7),
            top_p=gen_config.get('top_p', 0.9)
        )
        
        # 更新历史
        history.append({
            "user": message,
            "assistant": response
        })
        
        return response, history
    
    def batch_inference(self, prompts: List[str], **kwargs) -> List[str]:
        """批量推理"""
        results = []
        
        for prompt in prompts:
            try:
                response = self.generate_response(prompt, **kwargs)
                results.append(response)
            except Exception as e:
                self.logger.error(f"推理失败: {prompt[:50]}... - {e}")
                results.append(f"推理失败: {str(e)}")
        
        return results
    
    def security_demo(self) -> List[Dict[str, str]]:
        """网络安全功能演示"""
        demo_prompts = [
            "如何进行网络渗透测试？",
            "编写一个简单的端口扫描脚本",
            "解释SQL注入攻击的原理和防护方法",
            "生成一个简单的webshell代码",
            "如何检测和防护XSS攻击？",
            "编写一个网络数据包捕获程序"
        ]
        
        results = []
        
        for prompt in demo_prompts:
            self.logger.info(f"演示: {prompt}")
            response = self.generate_response(prompt)
            
            results.append({
                "prompt": prompt,
                "response": response
            })
            
            print(f"\n问题: {prompt}")
            print(f"回答: {response}")
            print("-" * 80)
        
        return results
    
    def interactive_chat(self) -> None:
        """交互式对话"""
        print("网络安全模型交互式对话")
        print("输入 'quit' 或 'exit' 退出")
        print("-" * 50)
        
        history = []
        
        while True:
            try:
                user_input = input("\n用户: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '退出']:
                    print("再见！")
                    break
                
                if not user_input:
                    continue
                
                response, history = self.chat(user_input, history)
                print(f"助手: {response}")
                
            except KeyboardInterrupt:
                print("\n再见！")
                break
            except Exception as e:
                print(f"错误: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        if self.model is None:
            return {"loaded": False}
        
        # 计算参数数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        info = {
            "loaded": True,
            "device": self.device,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_type": type(self.model).__name__,
            "tokenizer_vocab_size": len(self.tokenizer) if self.tokenizer else 0
        }
        
        # 检查是否是PEFT模型
        if hasattr(self.model, 'peft_config'):
            info["is_peft_model"] = True
            info["peft_type"] = str(type(self.model.peft_config))
        else:
            info["is_peft_model"] = False
        
        return info