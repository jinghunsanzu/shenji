#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试模型选择功能
"""

import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_download_model_selection():
    """测试下载模型时的选择功能"""
    print("=== 测试下载模型选择功能 ===")
    
    # 导入下载模块的函数
    sys.path.insert(0, os.path.dirname(__file__))
    from download_model import get_user_model_choice
    
    try:
        model_key = get_user_model_choice()
        print(f"\n选择的模型: {model_key}")
        
        # 获取模型配置信息
        from src.config.model_configs import ModelRegistry
        registry = ModelRegistry()
        model_config = registry.get_model_config(model_key)
        
        print(f"模型名称: {model_config.name}")
        print(f"模型ID: {model_config.model_id}")
        print(f"架构: {model_config.architecture}")
        
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        return False

def test_app_model_selection():
    """测试应用程序中的模型选择功能"""
    print("\n=== 测试应用程序模型选择功能 ===")
    
    try:
        from src.app import get_user_model_choice
        
        model_key = get_user_model_choice()
        print(f"\n选择的模型: {model_key}")
        
        # 设置模型
        from src.config.settings import Config
        Config.set_current_model(model_key)
        print(f"当前模型已设置为: {Config.get_current_model_key()}")
        
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        return False

def main():
    """主函数"""
    print("模型选择功能测试")
    print("=" * 50)
    
    # 测试1: 下载模型选择
    success1 = test_download_model_selection()
    
    # 测试2: 应用程序模型选择
    success2 = test_app_model_selection()
    
    if success1 and success2:
        print("\n✅ 所有测试通过！")
        return 0
    else:
        print("\n❌ 部分测试失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())