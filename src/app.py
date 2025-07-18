# -*- coding: utf-8 -*-
"""
主应用程序模块

整合所有功能模块，提供统一的训练接口。
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from .config import Config, TrainingConfig, DataConfig
from .data import DataDownloader, DataProcessor, DataLoader
from .model import ModelDownloader, SecurityModelTrainer, SecurityModelInference
from .utils import setup_logging, get_logger, EnvironmentChecker, TrainingMonitor


def get_user_model_choice():
    """获取用户的模型选择"""
    from .config.model_configs import ModelRegistry
    
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


class SecurityModelApp:
    """网络安全模型应用程序"""
    
    def __init__(self, config_path: Optional[str] = None):
        # 初始化配置
        self.config = Config()
        self.training_config = TrainingConfig()
        self.data_config = DataConfig()
        
        # 设置环境
        self.config.setup_environment()
        self.config.create_directories()
        
        # 设置日志
        setup_logging(self.config.LOGS_DIR)
        self.logger = get_logger(self.__class__.__name__)
        
        # 初始化组件
        self.env_checker = EnvironmentChecker()
        self.data_downloader = DataDownloader(self.config, self.data_config)
        self.data_processor = DataProcessor(self.config, self.data_config)
        self.data_loader = DataLoader(self.config, self.data_config)
        self.model_downloader = ModelDownloader(self.config)
        self.trainer = SecurityModelTrainer(self.config, self.training_config)
        self.monitor = TrainingMonitor(self.config.LOGS_DIR)
        
        self.logger.info("网络安全模型应用程序已初始化")
    
    def check_environment(self) -> bool:
        """检查环境"""
        self.logger.info("开始环境检查...")
        
        results = self.env_checker.check_all()
        
        # 检查关键项目
        critical_checks = ["python_version", "required_packages"]
        critical_failed = any(
            not results.get(check, {}).get("status", False)
            for check in critical_checks
        )
        
        if critical_failed:
            self.logger.error("关键环境检查失败，无法继续")
            recommendations = self.env_checker.get_recommendations()
            for rec in recommendations:
                self.logger.info(f"建议: {rec}")
            return False
        
        # 警告非关键项目
        non_critical_failed = [
            check for check, result in results.items()
            if check not in critical_checks and not result.get("status", False)
        ]
        
        if non_critical_failed:
            self.logger.warning(f"非关键检查失败: {', '.join(non_critical_failed)}")
            self.logger.warning("训练可能受到影响，但可以继续")
        
        return True
    
    def prepare_data(self, force_download: bool = False) -> bool:
        """准备训练数据"""
        try:
            self.logger.info("开始数据准备...")
            
            # 检查是否已有处理后的数据
            processed_data_file = self.config.DATA_DIR / "processed" / "training_data.json"
            
            if processed_data_file.exists() and not force_download:
                self.logger.info("发现已处理的数据，跳过数据下载和处理")
                return True
            
            # 下载数据
            self.logger.info("下载训练数据...")
            raw_data = self.data_downloader.download_all_data()
            
            if not raw_data:
                self.logger.error("数据下载失败")
                return False
            
            # 处理数据
            self.logger.info("处理训练数据...")
            clean_path, training_path = self.data_processor.process_and_save(raw_data)
            
            self.logger.info(f"数据处理完成: {training_path}")
            
            # 显示数据统计
            data_info = self.data_loader.get_data_info()
            self.logger.info(f"数据统计: {data_info}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"数据准备失败: {e}")
            return False
    
    def train_model(self, resume_from_checkpoint: bool = False, checkpoint_path: Optional[str] = None) -> bool:
        """训练模型"""
        try:
            if resume_from_checkpoint:
                if checkpoint_path:
                    self.logger.info(f"从指定checkpoint继续训练: {checkpoint_path}")
                else:
                    self.logger.info("从最新checkpoint继续训练")
            else:
                self.logger.info("开始新的模型训练...")
            
            # 开始监控
            self.monitor.start_monitoring()
            
            # 设置模型
            self.logger.info("设置模型...")
            self.trainer.setup_model()
            
            # 准备训练
            self.logger.info("准备训练数据...")
            self.trainer.prepare_training()
            
            # 显示训练信息
            training_info = self.trainer.get_training_info()
            self.logger.info(f"训练配置: {training_info}")
            
            # 开始训练（支持从checkpoint恢复）
            if resume_from_checkpoint:
                self.logger.info("开始恢复训练...")
                self.trainer.train(resume_from_checkpoint=checkpoint_path)
            else:
                self.logger.info("开始训练...")
                self.trainer.train()
            
            # 评估模型
            self.logger.info("评估模型...")
            eval_results = self.trainer.evaluate()
            self.logger.info(f"评估结果: {eval_results}")
            
            # 保存模型
            self.logger.info("保存模型...")
            model_path = self.trainer.save_model()
            self.logger.info(f"模型已保存到: {model_path}")
            
            # 测试模型
            self.logger.info("测试模型...")
            test_results = self.trainer.test_model()
            for result in test_results:
                self.logger.info(f"测试 - 问题: {result['prompt']}")
                self.logger.info(f"测试 - 回答: {result['response']}")
            
            # 停止监控并生成报告
            self.monitor.stop_monitoring()
            report_path = self.monitor.save_report()
            self.logger.info(f"训练报告已保存: {report_path}")
            
            self.logger.info("模型训练完成！")
            return True
            
        except Exception as e:
            self.logger.error(f"模型训练失败: {e}")
            self.monitor.stop_monitoring()
            return False
    
    def test_model(self, model_path: Optional[str] = None) -> bool:
        """测试模型"""
        try:
            self.logger.info("开始模型测试...")
            
            # 创建推理器
            inference = SecurityModelInference(self.config)
            
            # 加载模型
            if model_path:
                inference.load_model(model_path)
            else:
                # 使用默认输出路径
                default_path = self.config.OUTPUT_DIR / "final_model"
                if default_path.exists():
                    inference.load_model(str(default_path))
                else:
                    self.logger.error("未找到训练好的模型")
                    return False
            
            # 显示模型信息
            model_info = inference.get_model_info()
            self.logger.info(f"模型信息: {model_info}")
            
            # 运行安全功能演示
            self.logger.info("运行网络安全功能演示...")
            demo_results = inference.security_demo()
            
            self.logger.info("模型测试完成")
            return True
            
        except Exception as e:
            self.logger.error(f"模型测试失败: {e}")
            return False
    
    def interactive_mode(self, model_path: Optional[str] = None) -> None:
        """交互模式"""
        try:
            self.logger.info("启动交互模式...")
            
            # 创建推理器
            inference = SecurityModelInference(self.config)
            
            # 加载模型
            if model_path:
                inference.load_model(model_path)
            else:
                default_path = self.config.OUTPUT_DIR / "final_model"
                if default_path.exists():
                    inference.load_model(str(default_path))
                else:
                    self.logger.error("未找到训练好的模型")
                    return
            
            # 开始交互
            inference.interactive_chat()
            
        except Exception as e:
            self.logger.error(f"交互模式失败: {e}")
    
    def run_full_pipeline(self, force_data_download: bool = False, resume_from_checkpoint: bool = False, checkpoint_path: Optional[str] = None) -> bool:
        """运行完整流程"""
        self.logger.info("开始完整训练流程...")
        
        # 1. 环境检查
        if not self.check_environment():
            return False
        
        # 2. 数据准备
        if not self.prepare_data(force_data_download):
            return False
        
        # 3. 模型训练
        if not self.train_model(resume_from_checkpoint, checkpoint_path):
            return False
        
        # 4. 模型测试
        if not self.test_model():
            return False
        
        self.logger.info("完整训练流程成功完成！")
        return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="网络安全模型训练系统")
    parser.add_argument("--mode", choices=["full", "data", "train", "test", "interactive", "check"],
                       default="full", help="运行模式")
    parser.add_argument("--model-path", type=str, help="模型路径（用于测试和交互模式）")
    parser.add_argument("--force-download", action="store_true", help="强制重新下载数据")
    parser.add_argument("--config", type=str, help="配置文件路径")
    
    # 继续训练选项
    parser.add_argument("--resume", action="store_true", help="从最新checkpoint继续训练")
    parser.add_argument("--resume-from", type=str, help="从指定checkpoint路径继续训练")
    
    # 模型选择选项
    parser.add_argument("--model", type=str, help="选择基础模型 (qwen, chatglm, baichuan, llama等)")
    parser.add_argument("--list-models", action="store_true", help="列出支持的模型")
    
    args = parser.parse_args()
    
    try:
        # 处理列出模型选项
        if args.list_models:
            from .config.model_configs import ModelRegistry
            registry = ModelRegistry()
            models = registry.list_models()
            print("支持的模型:")
            for model_key, config in models.items():
                print(f"  {model_key}: {config.name} ({config.architecture})")
            return
        
        # 创建应用程序
        app = SecurityModelApp(args.config)
        
        # 设置模型（如果指定）
        if args.model:
            from .config.settings import Config
            try:
                Config.set_current_model(args.model)
                print(f"已切换到模型: {args.model}")
            except ValueError as e:
                print(f"模型设置失败: {e}")
                sys.exit(1)
        else:
            # 如果没有指定模型，提供交互式选择
            model_key = get_user_model_choice()
            from .config.settings import Config
            Config.set_current_model(model_key)
        
        # 处理继续训练选项
        resume_from_checkpoint = args.resume or args.resume_from
        checkpoint_path = args.resume_from if args.resume_from else None
        
        # 根据模式执行相应操作
        if args.mode == "check":
            success = app.check_environment()
        elif args.mode == "data":
            success = app.prepare_data(args.force_download)
        elif args.mode == "train":
            success = app.train_model(resume_from_checkpoint, checkpoint_path)
        elif args.mode == "test":
            success = app.test_model(args.model_path)
        elif args.mode == "interactive":
            app.interactive_mode(args.model_path)
            success = True
        elif args.mode == "full":
            success = app.run_full_pipeline(args.force_download, resume_from_checkpoint, checkpoint_path)
        else:
            print(f"未知模式: {args.mode}")
            success = False
        
        if not success:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"程序执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()