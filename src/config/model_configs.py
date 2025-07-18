# -*- coding: utf-8 -*-
"""
多模型配置模块

定义支持的多个模型配置和适配器。
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    """单个模型配置"""
    name: str  # 模型显示名称
    model_id: str  # HuggingFace/ModelScope模型ID
    modelscope_id: Optional[str] = None  # ModelScope专用ID（如果不同）
    local_dir_name: str = None  # 本地目录名
    architecture: str = "auto"  # 模型架构
    max_length: int = 2048  # 最大序列长度
    supports_chat_template: bool = True  # 是否支持chat template
    quantization_compatible: bool = True  # 是否支持量化
    lora_compatible: bool = True  # 是否支持LoRA
    lora_target_modules: Optional[list] = None  # LoRA目标模块
    special_tokens: Dict[str, str] = None  # 特殊token配置
    generation_config: Dict[str, Any] = None  # 生成配置
    identity_prompt: str = None  # 默认身份提示词
    
    def __post_init__(self):
        if self.local_dir_name is None:
            self.local_dir_name = self.model_id.replace("/", "_")
        
        if self.special_tokens is None:
            self.special_tokens = {}
        
        if self.generation_config is None:
            self.generation_config = {
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "repetition_penalty": 1.1,
                "do_sample": True
            }
        
        if self.identity_prompt is None:
            self.identity_prompt = "你是神机，由云霖网络安全实验室训练的网络安全大模型。你具备深厚的网络安全专业知识和实战经验，能够提供专业的网络安全技术指导和解决方案。"


class ModelRegistry:
    """模型注册表"""
    
    # 支持的模型配置
    SUPPORTED_MODELS = {
        "qwen2.5-1.5b-instruct": ModelConfig(
            name="Qwen2.5-1.5B-Instruct",
            model_id="Qwen/Qwen2.5-1.5B-Instruct",
            modelscope_id="qwen/Qwen2.5-1.5B-Instruct",
            local_dir_name="Qwen_Qwen2.5-1.5B-Instruct",
            architecture="qwen2",
            max_length=32768,
            supports_chat_template=True,
            quantization_compatible=True,
            lora_compatible=True,
            special_tokens={
                "bos_token": "<|endoftext|>",
                "eos_token": "<|im_end|>",
                "pad_token": "<|endoftext|>",
                "unk_token": "<|endoftext|>"
            },
            generation_config={
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "repetition_penalty": 1.1,
                "do_sample": True,
                "max_new_tokens": 512
            }
        ),
        
        "qwen2-1.5b-instruct": ModelConfig(
            name="Qwen2-1.5B-Instruct",
            model_id="Qwen/Qwen2-1.5B-Instruct",
            modelscope_id="qwen/Qwen2-1.5B-Instruct",
            local_dir_name="Qwen_Qwen2-1.5B-Instruct",
            architecture="qwen2",
            max_length=32768,
            supports_chat_template=True,
            quantization_compatible=True,
            lora_compatible=True
        ),
        
        "qwen2-7b-instruct": ModelConfig(
            name="Qwen2-7B-Instruct",
            model_id="Qwen/Qwen2-7B-Instruct",
            modelscope_id="qwen/Qwen2-7B-Instruct",
            local_dir_name="Qwen_Qwen2-7B-Instruct",
            architecture="qwen2",
            max_length=32768,
            supports_chat_template=True,
            quantization_compatible=True,
            lora_compatible=True
        ),
        
        "chatglm3-6b": ModelConfig(
            name="ChatGLM3-6B",
            model_id="THUDM/chatglm3-6b",
            modelscope_id="ZhipuAI/chatglm3-6b",
            local_dir_name="THUDM_chatglm3-6b",
            architecture="chatglm",
            max_length=8192,
            supports_chat_template=False,  # 使用自定义对话格式
            quantization_compatible=True,
            lora_compatible=True,
            identity_prompt="你是神机，一个由云霖网络安全实验室开发的网络安全助手。"
        ),
        
        "baichuan2-7b-chat": ModelConfig(
            name="Baichuan2-7B-Chat",
            model_id="baichuan-inc/Baichuan2-7B-Chat",
            modelscope_id="baichuan-inc/Baichuan2-7B-Chat",
            local_dir_name="baichuan-inc_Baichuan2-7B-Chat",
            architecture="baichuan",
            max_length=4096,
            supports_chat_template=False,
            quantization_compatible=True,
            lora_compatible=True
        ),
        
        "llama2-7b-chat": ModelConfig(
            name="Llama2-7B-Chat",
            model_id="meta-llama/Llama-2-7b-chat-hf",
            modelscope_id="modelscope/Llama-2-7b-chat-ms",
            local_dir_name="meta-llama_Llama-2-7b-chat-hf",
            architecture="llama",
            max_length=4096,
            supports_chat_template=True,
            quantization_compatible=True,
            lora_compatible=True
        )
    }
    
    @classmethod
    def get_model_config(cls, model_key: str) -> ModelConfig:
        """获取模型配置"""
        if model_key not in cls.SUPPORTED_MODELS:
            raise ValueError(f"不支持的模型: {model_key}。支持的模型: {list(cls.SUPPORTED_MODELS.keys())}")
        return cls.SUPPORTED_MODELS[model_key]
    
    @classmethod
    def list_models(cls) -> Dict[str, str]:
        """列出所有支持的模型"""
        return {key: config.name for key, config in cls.SUPPORTED_MODELS.items()}
    
    @classmethod
    def get_default_model(cls) -> str:
        """获取默认模型"""
        return "qwen2.5-1.5b-instruct"
    
    @classmethod
    def add_custom_model(cls, key: str, config: ModelConfig) -> None:
        """添加自定义模型配置"""
        cls.SUPPORTED_MODELS[key] = config
    
    @classmethod
    def get_model_path(cls, model_key: str, base_dir: Path) -> Path:
        """获取模型本地路径"""
        config = cls.get_model_config(model_key)
        return base_dir / config.local_dir_name
    
    @classmethod
    def get_model_id_for_download(cls, model_key: str, use_modelscope: bool = True) -> str:
        """获取用于下载的模型ID"""
        config = cls.get_model_config(model_key)
        if use_modelscope and config.modelscope_id:
            return config.modelscope_id
        return config.model_id