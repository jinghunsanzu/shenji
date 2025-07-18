# -*- coding: utf-8 -*-
"""
模型下载器模块

负责从ModelScope或HuggingFace下载模型。
"""

import os
from pathlib import Path
from typing import Optional

from ..config import Config
from ..utils.logger import get_logger


class ModelDownloader:
    """模型下载器"""
    
    def __init__(self, config: Config = None, model_key: str = None):
        self.config = config or Config()
        
        # 设置当前模型
        if model_key:
            self.config.set_current_model(model_key)
        
        # 获取模型配置
        self.model_config = self.config.get_current_model_config()
        self.logger = get_logger(self.__class__.__name__)
        
        # 设置环境变量
        self.config.setup_environment()
    
    def download_model(self, model_name: Optional[str] = None, 
                      force_download: bool = False) -> Path:
        """下载模型"""
        if model_name is None:
            model_name = self.config.get_current_model_key()
        
        model_path = self.get_model_path(model_name)
        
        # 检查模型是否已存在
        if self.check_model_exists(model_name) and not force_download:
            self.logger.info(f"模型已存在: {model_path}")
            return model_path
        
        self.logger.info(f"开始下载模型: {model_name}")
        
        try:
            if self.config.USE_MODELSCOPE:
                return self._download_from_modelscope(model_name, model_path)
            else:
                return self._download_from_huggingface(model_name, model_path)
        except Exception as e:
            self.logger.error(f"模型下载失败: {e}")
            # 如果ModelScope失败，尝试HuggingFace
            if self.config.USE_MODELSCOPE:
                self.logger.info("尝试从HuggingFace下载...")
                return self._download_from_huggingface(model_name, model_path)
            raise
    
    def _download_from_modelscope(self, model_name: str, model_path: Path) -> Path:
        """从ModelScope下载模型"""
        # 首先尝试git clone方式下载
        try:
            return self._download_from_modelscope_git(model_name, model_path)
        except Exception as git_error:
            self.logger.warning(f"Git下载失败: {git_error}，尝试SDK方式")
            
        # 如果git方式失败，回退到SDK方式
        try:
            return self._download_from_modelscope_sdk(model_name, model_path)
        except Exception as sdk_error:
            self.logger.error(f"SDK下载也失败: {sdk_error}")
            raise
    
    def _download_from_modelscope_git(self, model_name: str, model_path: Path) -> Path:
        """使用git clone从ModelScope下载模型"""
        import subprocess
        import shutil
        
        # 获取ModelScope下载ID
        download_id = self.config.get_model_id_for_download('modelscope', model_name)
        
        if not download_id:
            # 回退到原有映射逻辑
            modelscope_names = {
                "Qwen/Qwen2-1.5B": "qwen/Qwen2-1.5B",
                "Qwen/Qwen2-1.5B-Instruct": "qwen/Qwen2-1.5B-Instruct",
                "Qwen/Qwen2.5-1.5B-Instruct": "qwen/Qwen2.5-1.5B-Instruct"
            }
            download_id = modelscope_names.get(model_name, model_name)
        
        ms_model_name = download_id
        git_url = f"https://www.modelscope.cn/{ms_model_name}.git"
        
        self.logger.info(f"使用git clone从ModelScope下载: {git_url}")
        
        # 检查git是否可用
        try:
            subprocess.run(["git", "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise Exception("Git未安装或不可用")
        
        # 确保目标目录存在
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 如果目标目录已存在，先删除
        if model_path.exists():
            shutil.rmtree(model_path)
        
        # 执行git clone
        try:
            cmd = ["git", "clone", git_url, str(model_path)]
            result = subprocess.run(
                cmd, 
                check=True, 
                capture_output=True, 
                text=True,
                timeout=1800  # 30分钟超时
            )
            
            self.logger.info(f"Git clone成功: {model_path}")
            return model_path
            
        except subprocess.TimeoutExpired:
            raise Exception("Git clone超时")
        except subprocess.CalledProcessError as e:
            raise Exception(f"Git clone失败: {e.stderr}")
    
    def _download_from_modelscope_sdk(self, model_name: str, model_path: Path) -> Path:
        """使用SDK从ModelScope下载模型"""
        try:
            # 设置SSL配置以解决证书验证问题
            import ssl
            import urllib3
            import requests
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            
            # 彻底禁用SSL验证
            original_context = ssl._create_default_https_context
            ssl._create_default_https_context = ssl._create_unverified_context
            
            # Monkey patch urllib3 to disable SSL verification
            import urllib3.util.ssl_
            original_ssl_wrap_socket = urllib3.util.ssl_.ssl_wrap_socket
            
            def patched_ssl_wrap_socket(*args, **kwargs):
                kwargs['cert_reqs'] = ssl.CERT_NONE
                kwargs['check_hostname'] = False
                return original_ssl_wrap_socket(*args, **kwargs)
            
            urllib3.util.ssl_.ssl_wrap_socket = patched_ssl_wrap_socket
            
            try:
                from modelscope import snapshot_download
                
                # 获取ModelScope下载ID
                download_id = self.config.get_model_id_for_download('modelscope', model_name)
                
                if not download_id:
                    # 回退到原有映射逻辑
                    modelscope_names = {
                        "Qwen/Qwen2-1.5B": "qwen/Qwen2-1.5B",
                        "Qwen/Qwen2-1.5B-Instruct": "qwen/Qwen2-1.5B-Instruct",
                        "Qwen/Qwen2.5-1.5B-Instruct": "qwen/Qwen2.5-1.5B-Instruct"
                    }
                    download_id = modelscope_names.get(model_name, model_name)
                
                ms_model_name = download_id
                
                self.logger.info(f"使用SDK从ModelScope下载: {ms_model_name}")
                
                downloaded_path = snapshot_download(
                    model_id=ms_model_name,
                    cache_dir=str(self.config.MODELS_DIR),
                    local_dir=str(model_path)
                )
                
                self.logger.info(f"SDK下载完成: {downloaded_path}")
                return Path(downloaded_path)
                
            finally:
                # 恢复原始函数
                ssl._create_default_https_context = original_context
                urllib3.util.ssl_.ssl_wrap_socket = original_ssl_wrap_socket
            
        except ImportError:
            self.logger.error("ModelScope未安装，请安装: pip install modelscope")
            raise
    
    def _download_from_huggingface(self, model_name: str, model_path: Path) -> Path:
        """从HuggingFace下载模型"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            self.logger.info(f"从HuggingFace下载: {model_name}")
            
            # 下载分词器
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=str(self.config.CACHE_DIR),
                trust_remote_code=True
            )
            tokenizer.save_pretrained(str(model_path))
            
            # 下载模型
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=str(self.config.CACHE_DIR),
                trust_remote_code=True
            )
            model.save_pretrained(str(model_path))

            self.logger.info(f"模型下载完成: {model_path}")
            return model_path
            
        except Exception as e:
            self.logger.error(f"HuggingFace下载失败: {e}")
            raise
    
    def check_model_exists(self, model_name: Optional[str] = None) -> bool:
        """检查模型是否存在"""
        if model_name is None:
            model_name = self.config.get_current_model_key()
        
        model_path = self.get_model_path(model_name)
        
        # 检查关键文件是否存在
        required_files = ["config.json", "tokenizer.json"]
        
        if not model_path.exists():
            return False
        
        for file_name in required_files:
            if not (model_path / file_name).exists():
                return False
        
        return True
    
    def get_model_path(self, model_name: Optional[str] = None) -> Path:
        """获取模型路径"""
        if model_name is None:
            model_name = self.model_config.name
        
        # 尝试使用配置中的路径方法
        if hasattr(self.config, 'get_model_path'):
            return Path(self.config.get_model_path(model_name))
        
        # 优先检查原始名称（去掉组织前缀）
        simple_name = model_name.split("/")[-1] if "/" in model_name else model_name
        simple_path = self.config.MODELS_DIR / simple_name
        
        # 如果简单名称的目录存在，使用它
        if simple_path.exists():
            return simple_path
        
        # 否则使用下划线替换的名称
        return self.config.MODELS_DIR / model_name.replace("/", "_")
    
    def get_model_info(self, model_name: Optional[str] = None) -> dict:
        """获取模型信息"""
        model_path = self.get_model_path(model_name)
        
        if not self.check_model_exists(model_name):
            return {"exists": False, "path": str(model_path)}
        
        # 计算模型大小
        total_size = 0
        file_count = 0
        
        for file_path in model_path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
                file_count += 1
        
        return {
            "exists": True,
            "path": str(model_path),
            "size_mb": total_size / (1024 * 1024),
            "file_count": file_count
        }