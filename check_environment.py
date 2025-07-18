#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环境检查脚本
自动检查系统要求和依赖安装情况
"""

import sys
import subprocess
import importlib
from pathlib import Path


def check_python_version():
    """检查Python版本"""
    print("🔍 检查Python版本...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python版本: {version.major}.{version.minor}.{version.micro} (符合要求)")
        return True
    else:
        print(f"❌ Python版本: {version.major}.{version.minor}.{version.micro} (需要Python 3.8+)")
        return False


def check_memory():
    """检查系统内存"""
    print("\n🔍 检查系统内存...")
    try:
        import psutil
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        if memory_gb >= 8:
            print(f"✅ 系统内存: {memory_gb:.1f}GB (符合要求)")
            return True
        else:
            print(f"⚠️ 系统内存: {memory_gb:.1f}GB (推荐8GB+)")
            return True  # 不强制要求
    except ImportError:
        print("⚠️ 无法检查内存 (psutil未安装)")
        return True


def check_gpu():
    """检查GPU"""
    print("\n🔍 检查GPU...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ 检测到GPU: {gpu_name} (数量: {gpu_count})")
            return True
        else:
            print("⚠️ 未检测到可用GPU (可以使用CPU训练，但速度较慢)")
            return True
    except ImportError:
        print("⚠️ 无法检查GPU (PyTorch未安装)")
        return True


def check_dependencies():
    """检查核心依赖"""
    print("\n🔍 检查核心依赖...")
    
    required_packages = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'datasets': 'Datasets',
        'peft': 'PEFT',
        'accelerate': 'Accelerate',
        'huggingface_hub': 'Hugging Face Hub',
        'safetensors': 'SafeTensors',
        'sentencepiece': 'SentencePiece',
        'tqdm': 'TQDM',
        'requests': 'Requests',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'psutil': 'PSUtil'
    }
    
    missing_packages = []
    
    for package, name in required_packages.items():
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {name}: {version}")
        except ImportError:
            print(f"❌ {name}: 未安装")
            missing_packages.append(package)
    
    return len(missing_packages) == 0, missing_packages


def check_project_structure():
    """检查项目结构"""
    print("\n🔍 检查项目结构...")
    
    required_files = [
        'requirements.txt',
        'main.py',
        'start_training.sh',
        'src/app.py',
        'src/config/settings.py',
        'src/model/trainer.py',
        'src/data/loader.py'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}: 文件不存在")
            missing_files.append(file_path)
    
    return len(missing_files) == 0, missing_files


def provide_solutions(missing_packages, missing_files):
    """提供解决方案"""
    if missing_packages or missing_files:
        print("\n🔧 解决方案:")
        
        if missing_packages:
            print("\n📦 安装缺失的依赖:")
            print("pip install -r requirements.txt")
            print("\n或者逐个安装:")
            for package in missing_packages:
                print(f"pip install {package}")
        
        if missing_files:
            print("\n📁 缺失的文件:")
            for file_path in missing_files:
                print(f"- {file_path}")
            print("请确保您在正确的项目目录中运行此脚本")


def main():
    """主函数"""
    print("🚀 Qwen安全模型训练项目 - 环境检查")
    print("=" * 50)
    
    checks = [
        check_python_version(),
        check_memory(),
        check_gpu()
    ]
    
    deps_ok, missing_packages = check_dependencies()
    checks.append(deps_ok)
    
    structure_ok, missing_files = check_project_structure()
    checks.append(structure_ok)
    
    print("\n" + "=" * 50)
    
    if all(checks):
        print("🎉 环境检查通过！您可以开始使用项目了。")
        print("\n📚 下一步:")
        print("1. 运行: python main.py --help")
        print("2. 或者: ./start_training.sh --help")
        print("3. 查看: README.md 了解详细使用方法")
    else:
        print("⚠️ 环境检查发现问题，请按照以下建议解决:")
        provide_solutions(missing_packages, missing_files)
        print("\n解决问题后，请重新运行此脚本进行检查。")
    
    return all(checks)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)