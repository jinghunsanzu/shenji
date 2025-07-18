# -*- coding: utf-8 -*-
"""
主配置文件

定义系统的核心配置参数。
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from .model_configs import ModelRegistry


class Config:
    """系统主配置类"""
    
    # 项目根目录
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    
    # 基础路径配置
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    LOGS_DIR = PROJECT_ROOT / "logs"
    CACHE_DIR = PROJECT_ROOT / "cache"
    CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
    OUTPUT_DIR = PROJECT_ROOT / "output"
    
    # 模型配置
    DEFAULT_MODEL_KEY = "qwen2.5-1.5b-instruct"  # 默认模型
    CURRENT_MODEL_KEY = None  # 当前选择的模型，None表示使用默认
    MODEL_SAVE_PATH = OUTPUT_DIR / "trained_model"
    
    @classmethod
    def get_current_model_key(cls) -> str:
        """获取当前模型键"""
        return cls.CURRENT_MODEL_KEY or cls.DEFAULT_MODEL_KEY
    
    @classmethod
    def set_current_model(cls, model_key: str) -> None:
        """设置当前模型"""
        # 验证模型是否支持
        ModelRegistry.get_model_config(model_key)
        cls.CURRENT_MODEL_KEY = model_key
    
    @classmethod
    def get_current_model_config(cls):
        """获取当前模型配置"""
        return ModelRegistry.get_model_config(cls.get_current_model_key())
    
    @classmethod
    def get_model_path(cls, model_key: Optional[str] = None) -> Path:
        """获取模型本地路径"""
        if model_key is None:
            model_key = cls.get_current_model_key()
        return ModelRegistry.get_model_path(model_key, cls.MODELS_DIR)
    
    @classmethod
    def get_model_id_for_download(cls, platform: str = 'modelscope', model_key: Optional[str] = None) -> str:
        """获取用于下载的模型ID"""
        if model_key is None:
            model_key = cls.get_current_model_key()
        use_modelscope = platform == 'modelscope' if platform else cls.USE_MODELSCOPE
        return ModelRegistry.get_model_id_for_download(model_key, use_modelscope)
    
    # 兼容性：保持旧的BASE_MODEL_NAME属性
    @property
    def BASE_MODEL_NAME(cls) -> str:
        """兼容性属性：获取当前模型的下载ID"""
        return cls.get_model_id_for_download()
    
    # 系统配置
    USE_MODELSCOPE = True  # 使用国内源
    DEVICE = "cuda" if os.path.exists("/dev/nvidia0") else "cpu"
    
    # 环境变量
    HF_HOME = str(CACHE_DIR / "huggingface")
    TRANSFORMERS_CACHE = str(CACHE_DIR / "transformers")
    
    @classmethod
    def create_directories(cls) -> None:
        """创建必要的目录"""
        directories = [
            cls.DATA_DIR,
            cls.MODELS_DIR,
            cls.LOGS_DIR,
            cls.CACHE_DIR,
            cls.CHECKPOINTS_DIR,
            cls.OUTPUT_DIR,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def setup_environment(cls) -> None:
        """设置环境变量"""
        os.environ["HF_HOME"] = cls.HF_HOME
        os.environ["TRANSFORMERS_CACHE"] = cls.TRANSFORMERS_CACHE
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        if cls.DEVICE == "cuda":
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    @classmethod
    def get_log_file(cls, name: str) -> Path:
        """获取日志文件路径"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return cls.LOGS_DIR / f"{name}_{timestamp}.log"