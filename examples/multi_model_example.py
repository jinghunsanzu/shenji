#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模型使用示例

本示例展示如何在项目中使用不同的模型进行推理和训练。
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.settings import Config
from src.config.model_configs import ModelRegistry
from src.model.inference import SecurityModelInference
from src.model.trainer import SecurityModelTrainer
from src.model.downloader import ModelDownloader

def list_available_models():
    """列出所有可用的模型"""
    registry = ModelRegistry()
    models = registry.list_models()
    
    print("=== 可用模型列表 ===")
    for model_key, model_config in models.items():
        print(f"模型键: {model_key}")
        print(f"  名称: {model_config.name}")
        print(f"  架构: {model_config.architecture}")
        print(f"  最大长度: {model_config.max_length}")
        print(f"  支持Chat模板: {model_config.supports_chat_template}")
        print(f"  支持量化: {model_config.supports_quantization}")
        print(f"  支持LoRA: {model_config.supports_lora}")
        print()

def test_model_inference(model_key: str):
    """测试指定模型的推理功能"""
    print(f"\n=== 测试模型推理: {model_key} ===")
    
    try:
        # 创建配置和推理实例
        config = Config()
        inference = SecurityModelInference(config, model_key=model_key)
        
        # 检查模型是否已下载
        downloader = ModelDownloader(config, model_key=model_key)
        model_path = downloader.ensure_model_downloaded()
        print(f"模型路径: {model_path}")
        
        # 加载模型
        inference.load_model(model_key=model_key)
        print("模型加载成功")
        
        # 测试对话
        test_message = "请介绍一下网络安全的基本概念"
        response, history = inference.chat(test_message)
        
        print(f"用户: {test_message}")
        print(f"助手: {response}")
        
        return True
        
    except Exception as e:
        print(f"模型 {model_key} 测试失败: {e}")
        return False

def compare_models():
    """比较不同模型的回复"""
    print("\n=== 模型回复比较 ===")
    
    test_question = "什么是SQL注入攻击？如何防护？"
    print(f"测试问题: {test_question}\n")
    
    registry = ModelRegistry()
    models = registry.list_models()
    
    for model_key in models.keys():
        print(f"--- {model_key} ---")
        try:
            config = Config()
            inference = SecurityModelInference(config, model_key=model_key)
            
            # 检查模型是否存在
            model_path = Path(config.get_model_path())
            if not model_path.exists():
                print(f"模型未下载，跳过: {model_path}")
                continue
            
            inference.load_model(model_key=model_key)
            response, _ = inference.chat(test_question)
            
            print(f"回复: {response[:200]}..." if len(response) > 200 else f"回复: {response}")
            
        except Exception as e:
            print(f"错误: {e}")
        
        print()

def switch_model_demo():
    """演示模型切换功能"""
    print("\n=== 模型切换演示 ===")
    
    config = Config()
    
    # 显示当前模型
    current_model = config.get_current_model_key()
    print(f"当前模型: {current_model}")
    
    # 切换到不同的模型
    registry = ModelRegistry()
    models = list(registry.list_models().keys())
    
    for model_key in models[:3]:  # 只测试前3个模型
        print(f"\n切换到模型: {model_key}")
        config.set_current_model(model_key)
        
        current_config = config.get_current_model_config()
        print(f"模型名称: {current_config.name}")
        print(f"模型架构: {current_config.architecture}")
        print(f"本地路径: {config.get_model_path()}")

def download_model_demo(model_key: str):
    """演示模型下载功能"""
    print(f"\n=== 下载模型演示: {model_key} ===")
    
    try:
        config = Config()
        downloader = ModelDownloader(config, model_key=model_key)
        
        # 获取下载信息
        modelscope_id = config.get_model_id_for_download('modelscope')
        huggingface_id = config.get_model_id_for_download('huggingface')
        
        print(f"ModelScope ID: {modelscope_id}")
        print(f"HuggingFace ID: {huggingface_id}")
        print(f"本地路径: {config.get_model_path()}")
        
        # 检查是否已下载
        model_path = Path(config.get_model_path())
        if model_path.exists() and any(model_path.iterdir()):
            print("模型已存在")
        else:
            print("模型需要下载")
            # 注意：实际下载可能需要很长时间，这里只是演示
            # model_path = downloader.ensure_model_downloaded()
            # print(f"下载完成: {model_path}")
        
    except Exception as e:
        print(f"下载演示失败: {e}")

def main():
    """主函数"""
    print("神机多模型支持演示")
    print("=" * 50)
    
    # 1. 列出可用模型
    list_available_models()
    
    # 2. 演示模型切换
    switch_model_demo()
    
    # 3. 演示下载功能
    download_model_demo('qwen2.5-1.5b')
    download_model_demo('chatglm3-6b')
    
    # 4. 测试模型推理（如果模型已下载）
    registry = ModelRegistry()
    for model_key in list(registry.list_models().keys())[:2]:  # 只测试前2个
        config = Config()
        config.set_current_model(model_key)
        model_path = Path(config.get_model_path())
        
        if model_path.exists() and any(model_path.iterdir()):
            test_model_inference(model_key)
        else:
            print(f"\n模型 {model_key} 未下载，跳过推理测试")
    
    # 5. 比较模型回复（如果有多个模型已下载）
    # compare_models()
    
    print("\n演示完成！")

if __name__ == "__main__":
    main()