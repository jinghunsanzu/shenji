#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型管理CLI工具

提供命令行界面来管理多个模型，包括列出、下载、切换和测试模型。

使用方法:
    python scripts/model_manager.py list                    # 列出所有可用模型
    python scripts/model_manager.py download <model_key>    # 下载指定模型
    python scripts/model_manager.py switch <model_key>      # 切换当前模型
    python scripts/model_manager.py current                 # 显示当前模型
    python scripts/model_manager.py test <model_key>        # 测试模型推理
    python scripts/model_manager.py chat <model_key>        # 与模型对话
    python scripts/model_manager.py info <model_key>        # 显示模型详细信息
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.settings import Config
from src.config.model_configs import ModelRegistry
from src.model.inference import SecurityModelInference
from src.model.downloader import ModelDownloader
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ModelManager:
    """模型管理器"""
    
    def __init__(self):
        self.config = Config()
        self.registry = ModelRegistry()
    
    def list_models(self):
        """列出所有可用模型"""
        models = self.registry.list_models()
        
        print("\n=== 可用模型列表 ===")
        print(f"{'模型键':<20} {'名称':<30} {'架构':<15} {'状态':<10}")
        print("-" * 80)
        
        current_model = self.config.get_current_model_key()
        
        for model_key, model_config in models.items():
            # 检查模型是否已下载
            self.config.set_current_model(model_key)
            model_path = Path(self.config.get_model_path())
            status = "已下载" if model_path.exists() and any(model_path.iterdir()) else "未下载"
            
            # 标记当前模型
            marker = "*" if model_key == current_model else " "
            
            print(f"{marker}{model_key:<19} {model_config.name:<30} {model_config.architecture:<15} {status:<10}")
        
        print(f"\n当前模型: {current_model}")
        print("注: * 表示当前选中的模型")
    
    def download_model(self, model_key: str):
        """下载指定模型"""
        if model_key not in self.registry.list_models():
            print(f"错误: 未知的模型键 '{model_key}'")
            self.list_available_keys()
            return False
        
        try:
            print(f"\n开始下载模型: {model_key}")
            
            downloader = ModelDownloader(self.config, model_key=model_key)
            model_path = downloader.ensure_model_downloaded()
            
            print(f"模型下载成功: {model_path}")
            return True
            
        except Exception as e:
            print(f"模型下载失败: {e}")
            logger.error(f"下载模型 {model_key} 失败", exc_info=True)
            return False
    
    def switch_model(self, model_key: str):
        """切换当前模型"""
        if model_key not in self.registry.list_models():
            print(f"错误: 未知的模型键 '{model_key}'")
            self.list_available_keys()
            return False
        
        try:
            old_model = self.config.get_current_model_key()
            self.config.set_current_model(model_key)
            
            print(f"模型切换成功: {old_model} -> {model_key}")
            return True
            
        except Exception as e:
            print(f"模型切换失败: {e}")
            logger.error(f"切换到模型 {model_key} 失败", exc_info=True)
            return False
    
    def show_current(self):
        """显示当前模型信息"""
        current_key = self.config.get_current_model_key()
        current_config = self.config.get_current_model_config()
        
        print(f"\n=== 当前模型信息 ===")
        print(f"模型键: {current_key}")
        print(f"名称: {current_config.name}")
        print(f"架构: {current_config.architecture}")
        print(f"最大长度: {current_config.max_length}")
        print(f"本地路径: {self.config.get_model_path()}")
        
        # 检查模型状态
        model_path = Path(self.config.get_model_path())
        if model_path.exists() and any(model_path.iterdir()):
            print(f"状态: 已下载")
        else:
            print(f"状态: 未下载")
    
    def show_model_info(self, model_key: str):
        """显示指定模型的详细信息"""
        if model_key not in self.registry.list_models():
            print(f"错误: 未知的模型键 '{model_key}'")
            self.list_available_keys()
            return False
        
        model_config = self.registry.get_model_config(model_key)
        
        print(f"\n=== 模型信息: {model_key} ===")
        print(f"名称: {model_config.name}")
        print(f"架构: {model_config.architecture}")
        print(f"最大长度: {model_config.max_length}")
        print(f"支持Chat模板: {model_config.supports_chat_template}")
        print(f"支持量化: {model_config.supports_quantization}")
        print(f"支持LoRA: {model_config.supports_lora}")
        
        # 下载信息
        self.config.set_current_model(model_key)
        modelscope_id = self.config.get_model_id_for_download('modelscope')
        huggingface_id = self.config.get_model_id_for_download('huggingface')
        
        print(f"\n下载信息:")
        print(f"  ModelScope: {modelscope_id or '不支持'}")
        print(f"  HuggingFace: {huggingface_id or '不支持'}")
        print(f"  本地路径: {self.config.get_model_path()}")
        
        # 检查状态
        model_path = Path(self.config.get_model_path())
        if model_path.exists() and any(model_path.iterdir()):
            print(f"  状态: 已下载")
        else:
            print(f"  状态: 未下载")
        
        # 特殊配置
        if model_config.special_tokens:
            print(f"\n特殊Token: {model_config.special_tokens}")
        
        if model_config.lora_target_modules:
            print(f"LoRA目标模块: {model_config.lora_target_modules}")
        
        if model_config.generation_config:
            print(f"生成配置: {model_config.generation_config}")
        
        return True
    
    def test_model(self, model_key: str):
        """测试模型推理"""
        if model_key not in self.registry.list_models():
            print(f"错误: 未知的模型键 '{model_key}'")
            self.list_available_keys()
            return False
        
        try:
            print(f"\n测试模型: {model_key}")
            
            # 检查模型是否已下载
            self.config.set_current_model(model_key)
            model_path = Path(self.config.get_model_path())
            
            if not model_path.exists() or not any(model_path.iterdir()):
                print(f"模型未下载，请先下载: python {sys.argv[0]} download {model_key}")
                return False
            
            # 创建推理实例
            inference = SecurityModelInference(self.config, model_key=model_key)
            inference.load_model(model_key=model_key)
            
            # 测试对话
            test_message = "请简单介绍一下你自己"
            print(f"\n测试问题: {test_message}")
            print("生成回复中...")
            
            response, _ = inference.chat(test_message)
            
            print(f"\n模型回复: {response}")
            print(f"\n测试完成！模型 {model_key} 工作正常。")
            return True
            
        except Exception as e:
            print(f"模型测试失败: {e}")
            logger.error(f"测试模型 {model_key} 失败", exc_info=True)
            return False
    
    def chat_with_model(self, model_key: str):
        """与模型进行交互式对话"""
        if model_key not in self.registry.list_models():
            print(f"错误: 未知的模型键 '{model_key}'")
            self.list_available_keys()
            return False
        
        try:
            print(f"\n启动与模型 {model_key} 的对话")
            
            # 检查模型是否已下载
            self.config.set_current_model(model_key)
            model_path = Path(self.config.get_model_path())
            
            if not model_path.exists() or not any(model_path.iterdir()):
                print(f"模型未下载，请先下载: python {sys.argv[0]} download {model_key}")
                return False
            
            # 创建推理实例
            print("加载模型中...")
            inference = SecurityModelInference(self.config, model_key=model_key)
            inference.load_model(model_key=model_key)
            
            print("模型加载完成！")
            print("输入 'quit' 或 'exit' 退出对话")
            print("-" * 50)
            
            history = []
            
            while True:
                try:
                    user_input = input("\n用户: ").strip()
                    
                    if user_input.lower() in ['quit', 'exit', '退出']:
                        print("对话结束")
                        break
                    
                    if not user_input:
                        continue
                    
                    print("助手: ", end="", flush=True)
                    response, history = inference.chat(user_input, history)
                    print(response)
                    
                except KeyboardInterrupt:
                    print("\n\n对话被中断")
                    break
                except Exception as e:
                    print(f"\n对话出错: {e}")
                    continue
            
            return True
            
        except Exception as e:
            print(f"启动对话失败: {e}")
            logger.error(f"与模型 {model_key} 对话失败", exc_info=True)
            return False
    
    def list_available_keys(self):
        """列出可用的模型键"""
        models = self.registry.list_models()
        print(f"\n可用的模型键: {', '.join(models.keys())}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="神机模型管理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s list                    # 列出所有模型
  %(prog)s download qwen2.5-1.5b  # 下载Qwen模型
  %(prog)s switch chatglm3-6b     # 切换到ChatGLM模型
  %(prog)s current                 # 显示当前模型
  %(prog)s test qwen2.5-1.5b       # 测试模型
  %(prog)s chat qwen2.5-1.5b       # 与模型对话
  %(prog)s info baichuan2-7b       # 显示模型信息
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # list命令
    subparsers.add_parser('list', help='列出所有可用模型')
    
    # download命令
    download_parser = subparsers.add_parser('download', help='下载指定模型')
    download_parser.add_argument('model_key', help='模型键')
    
    # switch命令
    switch_parser = subparsers.add_parser('switch', help='切换当前模型')
    switch_parser.add_argument('model_key', help='模型键')
    
    # current命令
    subparsers.add_parser('current', help='显示当前模型信息')
    
    # test命令
    test_parser = subparsers.add_parser('test', help='测试模型推理')
    test_parser.add_argument('model_key', help='模型键')
    
    # chat命令
    chat_parser = subparsers.add_parser('chat', help='与模型进行交互式对话')
    chat_parser.add_argument('model_key', help='模型键')
    
    # info命令
    info_parser = subparsers.add_parser('info', help='显示模型详细信息')
    info_parser.add_argument('model_key', help='模型键')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = ModelManager()
    
    try:
        if args.command == 'list':
            manager.list_models()
        
        elif args.command == 'download':
            success = manager.download_model(args.model_key)
            sys.exit(0 if success else 1)
        
        elif args.command == 'switch':
            success = manager.switch_model(args.model_key)
            sys.exit(0 if success else 1)
        
        elif args.command == 'current':
            manager.show_current()
        
        elif args.command == 'test':
            success = manager.test_model(args.model_key)
            sys.exit(0 if success else 1)
        
        elif args.command == 'chat':
            success = manager.chat_with_model(args.model_key)
            sys.exit(0 if success else 1)
        
        elif args.command == 'info':
            success = manager.show_model_info(args.model_key)
            sys.exit(0 if success else 1)
        
        else:
            print(f"未知命令: {args.command}")
            parser.print_help()
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n操作被中断")
        sys.exit(1)
    except Exception as e:
        print(f"执行失败: {e}")
        logger.error("命令执行失败", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()