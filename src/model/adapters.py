# -*- coding: utf-8 -*-
"""
模型适配器模块

为不同模型提供统一的接口和特殊处理逻辑。
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from ..config.model_configs import ModelConfig
from ..utils.logger import get_logger


class BaseModelAdapter(ABC):
    """模型适配器基类"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
    
    @abstractmethod
    def format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """格式化对话提示词"""
        pass
    
    @abstractmethod
    def setup_tokenizer(self, tokenizer: AutoTokenizer) -> AutoTokenizer:
        """设置分词器特殊配置"""
        pass
    
    @abstractmethod
    def setup_model(self, model: AutoModelForCausalLM) -> AutoModelForCausalLM:
        """设置模型特殊配置"""
        pass
    
    def get_generation_config(self) -> Dict[str, Any]:
        """获取生成配置"""
        return self.config.generation_config.copy()
    
    def get_default_system_message(self) -> str:
        """获取默认系统消息"""
        return self.config.identity_prompt


class QwenAdapter(BaseModelAdapter):
    """Qwen模型适配器"""
    
    def format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """使用Qwen的chat template格式化对话"""
        # Qwen支持标准的chat template，直接返回None让tokenizer处理
        return None
    
    def setup_tokenizer(self, tokenizer: AutoTokenizer) -> AutoTokenizer:
        """设置Qwen分词器"""
        # 设置特殊token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 设置padding方向
        tokenizer.padding_side = "right"
        
        return tokenizer
    
    def setup_model(self, model: AutoModelForCausalLM) -> AutoModelForCausalLM:
        """设置Qwen模型"""
        # Qwen模型通常不需要特殊设置
        return model


class ChatGLMAdapter(BaseModelAdapter):
    """ChatGLM模型适配器"""
    
    def format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """使用ChatGLM的对话格式"""
        prompt = ""
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                prompt += f"[系统] {content}\n"
            elif role == "user":
                prompt += f"[用户] {content}\n"
            elif role == "assistant":
                prompt += f"[助手] {content}\n"
        
        prompt += "[助手] "
        return prompt
    
    def setup_tokenizer(self, tokenizer: AutoTokenizer) -> AutoTokenizer:
        """设置ChatGLM分词器"""
        # ChatGLM有自己的特殊token设置
        tokenizer.padding_side = "left"  # ChatGLM通常使用左填充
        return tokenizer
    
    def setup_model(self, model: AutoModelForCausalLM) -> AutoModelForCausalLM:
        """设置ChatGLM模型"""
        # 启用ChatGLM的特殊模式
        if hasattr(model, 'transformer'):
            model.transformer.output_hidden_states = False
        return model


class BaichuanAdapter(BaseModelAdapter):
    """Baichuan模型适配器"""
    
    def format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """使用Baichuan的对话格式"""
        prompt = ""
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                prompt += f"<reserved_102>{content}<reserved_103>"
            elif role == "user":
                prompt += f"<reserved_106>{content}<reserved_107>"
            elif role == "assistant":
                prompt += f"{content}"
        
        return prompt
    
    def setup_tokenizer(self, tokenizer: AutoTokenizer) -> AutoTokenizer:
        """设置Baichuan分词器"""
        tokenizer.padding_side = "right"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    
    def setup_model(self, model: AutoModelForCausalLM) -> AutoModelForCausalLM:
        """设置Baichuan模型"""
        return model


class LlamaAdapter(BaseModelAdapter):
    """Llama模型适配器"""
    
    def format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """使用Llama的对话格式"""
        # Llama2-Chat使用特殊的格式
        prompt = ""
        
        system_message = None
        conversation = []
        
        for message in messages:
            if message["role"] == "system":
                system_message = message["content"]
            else:
                conversation.append(message)
        
        # 构建Llama2-Chat格式
        if system_message:
            prompt += f"<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n"
        else:
            prompt += "<s>[INST] "
        
        for i, message in enumerate(conversation):
            if message["role"] == "user":
                if i == 0 and system_message:
                    prompt += f"{message['content']} [/INST]"
                else:
                    prompt += f"<s>[INST] {message['content']} [/INST]"
            elif message["role"] == "assistant":
                prompt += f" {message['content']} </s>"
        
        return prompt
    
    def setup_tokenizer(self, tokenizer: AutoTokenizer) -> AutoTokenizer:
        """设置Llama分词器"""
        tokenizer.padding_side = "right"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token or "<unk>"
        return tokenizer
    
    def setup_model(self, model: AutoModelForCausalLM) -> AutoModelForCausalLM:
        """设置Llama模型"""
        return model


class ModelAdapterFactory:
    """模型适配器工厂"""
    
    _adapters = {
        "qwen2": QwenAdapter,
        "chatglm": ChatGLMAdapter,
        "baichuan": BaichuanAdapter,
        "llama": LlamaAdapter,
    }
    
    @classmethod
    def create_adapter(cls, config: ModelConfig) -> BaseModelAdapter:
        """创建模型适配器"""
        architecture = config.architecture.lower()
        
        # 处理架构名称的变体
        if architecture.startswith("qwen"):
            architecture = "qwen2"
        elif architecture.startswith("chatglm"):
            architecture = "chatglm"
        elif architecture.startswith("baichuan"):
            architecture = "baichuan"
        elif architecture.startswith("llama"):
            architecture = "llama"
        
        if architecture not in cls._adapters:
            # 如果没有专门的适配器，使用Qwen适配器作为默认
            architecture = "qwen2"
        
        adapter_class = cls._adapters[architecture]
        return adapter_class(config)
    
    @classmethod
    def register_adapter(cls, architecture: str, adapter_class: type) -> None:
        """注册自定义适配器"""
        cls._adapters[architecture] = adapter_class