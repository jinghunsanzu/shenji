# -*- coding: utf-8 -*-
"""
训练配置模块

定义模型训练相关的所有参数。
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TrainingConfig:
    """训练配置类"""
    
    # 基础训练参数
    batch_size: int = 1  # 适配6GB显存
    gradient_accumulation_steps: int = 16  # 通过梯度累积增加有效batch size
    learning_rate: float = 2e-4
    num_epochs: int = 3
    max_length: int = 256  # 减小序列长度节省显存
    warmup_steps: int = 100
    save_steps: int = 500
    logging_steps: int = 50
    eval_steps: int = 500
    
    # LoRA配置
    use_lora: bool = True  # 是否使用LoRA微调
    lora_config: Optional['LoRAConfig'] = None
    
    # 优化器配置
    optim: str = "adamw_torch"
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01
    
    # 精度和性能配置
    fp16: bool = True  # 使用半精度节省显存
    gradient_checkpointing: bool = False  # 暂时禁用梯度检查点避免冲突
    dataloader_num_workers: int = 0  # 减少worker数量节省内存
    remove_unused_columns: bool = False
    
    # 模型保存配置
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # 数据分割
    train_test_split: float = 0.95
    
    # 随机种子
    seed: int = 42
    
    def __post_init__(self):
        if self.lora_config is None:
            self.lora_config = LoRAConfig()
    
    def to_training_arguments_dict(self) -> dict:
        """转换为TrainingArguments所需的字典格式"""
        return {
            "per_device_train_batch_size": self.batch_size,
            "per_device_eval_batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "num_train_epochs": self.num_epochs,
            "warmup_steps": self.warmup_steps,
            "logging_steps": self.logging_steps,
            "save_steps": self.save_steps,
            "eval_steps": self.eval_steps,
            "eval_strategy": "steps",
            "save_strategy": "steps",
            "load_best_model_at_end": self.load_best_model_at_end,
            "metric_for_best_model": self.metric_for_best_model,
            "greater_is_better": self.greater_is_better,
            "fp16": self.fp16,
            "gradient_checkpointing": self.gradient_checkpointing,
            "dataloader_num_workers": self.dataloader_num_workers,
            "remove_unused_columns": self.remove_unused_columns,
            "report_to": None,
            "save_total_limit": self.save_total_limit,
            "prediction_loss_only": True,
            "optim": self.optim,
            "lr_scheduler_type": self.lr_scheduler_type,
            "weight_decay": self.weight_decay,
            "seed": self.seed,
            "overwrite_output_dir": True,
        }


@dataclass
class LoRAConfig:
    """LoRA配置类"""
    
    r: int = 16  # LoRA秩
    lora_alpha: int = 32  # LoRA alpha参数
    lora_dropout: float = 0.1  # LoRA dropout
    bias: str = "none"  # bias类型
    target_modules: List[str] = None  # 目标模块
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
    
    def to_peft_config_dict(self) -> dict:
        """转换为PEFT配置字典"""
        return {
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "bias": self.bias,
            "target_modules": self.target_modules,
        }