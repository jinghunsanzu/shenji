#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型下载脚本

提供多种下载方式和错误处理机制
"""

import os
import sys
import ssl
import subprocess
from pathlib import Path
from typing import Optional


def setup_ssl_bypass():
    """设置SSL绕过配置"""
    # 禁用SSL验证
    ssl._create_default_https_context = ssl._create_unverified_context
    
    # 设置环境变量
    os.environ['PYTHONHTTPSVERIFY'] = '0'
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['REQUESTS_CA_BUNDLE'] = ''
    
    # 禁用urllib3警告
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def download_with_modelscope_cli(model_name: str, output_dir: str) -> bool:
    """使用ModelScope命令行工具下载"""
    try:
        print(f"尝试使用ModelScope CLI下载: {model_name}")
        
        # 设置环境变量
        env = os.environ.copy()
        env.update({
            'PYTHONHTTPSVERIFY': '0',
            'SSL_VERIFY': 'false',
            'CURL_CA_BUNDLE': '',
            'REQUESTS_CA_BUNDLE': ''
        })
        
        cmd = ['modelscope', 'download', '--model', model_name, '--local_dir', output_dir]
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✓ ModelScope CLI下载成功: {output_dir}")
            return True
        else:
            print(f"✗ ModelScope CLI下载失败: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"✗ ModelScope CLI下载异常: {e}")
        return False


def download_with_python_api(model_name: str, output_dir: str) -> bool:
    """使用Python API下载"""
    try:
        print(f"尝试使用Python API下载: {model_name}")
        
        from modelscope import snapshot_download
        
        # 模型名称映射
        modelscope_names = {
            "Qwen/Qwen2.5-1.5B-Instruct": "qwen/Qwen2.5-1.5B-Instruct",
            "Qwen/Qwen2-1.5B": "qwen/Qwen2-1.5B",
            "Qwen/Qwen2-1.5B-Instruct": "qwen/Qwen2-1.5B-Instruct"
        }
        
        ms_model_name = modelscope_names.get(model_name, model_name)
        
        downloaded_path = snapshot_download(
            model_id=ms_model_name,
            local_dir=output_dir
        )
        
        print(f"✓ Python API下载成功: {downloaded_path}")
        return True
        
    except Exception as e:
        print(f"✗ Python API下载失败: {e}")
        return False


def verify_model_files(model_dir: str) -> bool:
    """验证模型文件完整性"""
    model_path = Path(model_dir)
    
    if not model_path.exists():
        print(f"✗ 模型目录不存在: {model_dir}")
        return False
    
    # 检查必要文件
    required_files = [
        'config.json',
        'tokenizer.json',
        'tokenizer_config.json'
    ]
    
    missing_files = []
    for file_name in required_files:
        if not (model_path / file_name).exists():
            missing_files.append(file_name)
    
    if missing_files:
        print(f"✗ 缺少必要文件: {missing_files}")
        return False
    
    # 检查是否只有临时文件夹
    contents = list(model_path.iterdir())
    if len(contents) == 1 and contents[0].name.startswith('._____temp'):
        print(f"✗ 模型目录只包含临时文件夹")
        return False
    
    print(f"✓ 模型文件验证通过")
    return True


def get_user_model_choice():
    """获取用户的模型选择"""
    from src.config.model_configs import ModelRegistry
    
    print("\n=== 模型选择 ===")
    print("可用的模型:")
    
    # 显示可用模型列表
    registry = ModelRegistry()
    models = registry.list_models()  # 返回 {key: name} 格式
    model_list = list(models.keys())
    
    for i, (model_key, model_name) in enumerate(models.items(), 1):
        # 获取完整配置以显示架构信息
        config = registry.get_model_config(model_key)
        print(f"  {i}. {model_key}: {model_name} ({config.architecture})")
    
    print(f"\n默认模型: Qwen2.5-1.5B-Instruct (qwen2.5-1.5b-instruct)")
    print("请选择模型 (输入数字编号，或直接回车使用默认模型):")
    
    try:
        user_input = input("> ").strip()
        
        if not user_input:  # 用户直接回车，使用默认模型
            return "qwen2.5-1.5b-instruct"
        
        # 尝试解析为数字
        try:
            choice_num = int(user_input)
            if 1 <= choice_num <= len(model_list):
                selected_key = model_list[choice_num - 1]
                print(f"已选择: {selected_key}")
                return selected_key
            else:
                print(f"无效的选择编号，使用默认模型")
                return "qwen2.5-1.5b-instruct"
        except ValueError:
            # 尝试直接匹配模型键
            if user_input in models:
                print(f"已选择: {user_input}")
                return user_input
            else:
                print(f"未找到模型 '{user_input}'，使用默认模型")
                return "qwen2.5-1.5b-instruct"
                
    except KeyboardInterrupt:
        print("\n用户取消，使用默认模型")
        return "qwen2.5-1.5b-instruct"
    except Exception as e:
        print(f"输入错误: {e}，使用默认模型")
        return "qwen2.5-1.5b-instruct"


def main():
    """主函数"""
    # 获取用户选择的模型
    model_key = get_user_model_choice()
    
    # 获取模型配置
    from src.config.model_configs import ModelRegistry
    registry = ModelRegistry()
    model_config = registry.get_model_config(model_key)
    
    model_name = model_config.model_id  # 用于下载的实际模型ID
    output_dir = f"/qwen/models/{model_config.name.replace('-', '_')}"
    
    print(f"开始下载模型: {model_name}")
    print(f"输出目录: {output_dir}")
    
    # 清理不完整的下载
    if Path(output_dir).exists():
        if not verify_model_files(output_dir):
            print(f"清理不完整的下载目录: {output_dir}")
            import shutil
            shutil.rmtree(output_dir)
    
    # 设置SSL绕过
    setup_ssl_bypass()
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 尝试多种下载方式
    success = False
    
    # 方式1: ModelScope CLI
    if not success:
        success = download_with_modelscope_cli(model_name, output_dir)
    
    # 方式2: Python API
    if not success:
        success = download_with_python_api(model_name, output_dir)
    
    # 验证下载结果
    if success:
        if verify_model_files(output_dir):
            print(f"\n🎉 模型下载成功！")
            print(f"模型路径: {output_dir}")
            return 0
        else:
            print(f"\n❌ 模型下载不完整")
            return 1
    else:
        print(f"\n❌ 所有下载方式都失败了")
        print(f"\n建议解决方案:")
        print(f"1. 检查网络连接")
        print(f"2. 配置代理服务器")
        print(f"3. 手动下载模型文件")
        return 1


if __name__ == "__main__":
    sys.exit(main())