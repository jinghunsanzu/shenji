# -*- coding: utf-8 -*-
"""
日志工具模块

提供统一的日志管理功能。
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logging(log_dir: Optional[Path] = None, log_level: str = "INFO") -> None:
    """设置日志配置"""
    if log_dir is None:
        log_dir = Path("logs")
    
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"
    
    # 配置日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 配置根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # 清除现有处理器
    root_logger.handlers.clear()
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 记录日志配置信息
    root_logger.info(f"日志系统已初始化，日志文件: {log_file}")


def get_logger(name: str) -> logging.Logger:
    """获取日志器"""
    return logging.getLogger(name)


class TrainingLogger:
    """训练日志器"""
    
    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = log_dir or Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = get_logger("TrainingLogger")
        
        # 训练指标日志文件
        self.metrics_file = self.log_dir / "training_metrics.log"
    
    def log_epoch_start(self, epoch: int, total_epochs: int) -> None:
        """记录训练轮次开始"""
        self.logger.info(f"开始第 {epoch}/{total_epochs} 轮训练")
    
    def log_epoch_end(self, epoch: int, metrics: dict) -> None:
        """记录训练轮次结束"""
        self.logger.info(f"第 {epoch} 轮训练完成")
        
        # 记录指标
        for key, value in metrics.items():
            self.logger.info(f"{key}: {value}")
        
        # 保存到指标文件
        self._save_metrics(epoch, metrics)
    
    def log_step(self, step: int, loss: float, lr: float = None) -> None:
        """记录训练步骤"""
        msg = f"Step {step}: loss={loss:.4f}"
        if lr is not None:
            msg += f", lr={lr:.2e}"
        
        self.logger.info(msg)
    
    def log_evaluation(self, metrics: dict) -> None:
        """记录评估结果"""
        self.logger.info("评估结果:")
        for key, value in metrics.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_model_info(self, info: dict) -> None:
        """记录模型信息"""
        self.logger.info("模型信息:")
        for key, value in info.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_data_info(self, info: dict) -> None:
        """记录数据信息"""
        self.logger.info("数据信息:")
        for key, value in info.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_error(self, error: Exception, context: str = "") -> None:
        """记录错误"""
        if context:
            self.logger.error(f"{context}: {str(error)}")
        else:
            self.logger.error(str(error))
        
        # 记录详细错误信息
        self.logger.exception("详细错误信息:")
    
    def _save_metrics(self, epoch: int, metrics: dict) -> None:
        """保存训练指标到文件"""
        timestamp = datetime.now().isoformat()
        
        with open(self.metrics_file, 'a', encoding='utf-8') as f:
            f.write(f"{timestamp},epoch={epoch}")
            for key, value in metrics.items():
                f.write(f",{key}={value}")
            f.write("\n")
    
    def get_latest_metrics(self, num_lines: int = 10) -> list:
        """获取最新的训练指标"""
        if not self.metrics_file.exists():
            return []
        
        with open(self.metrics_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        return lines[-num_lines:] if lines else []


class ProgressLogger:
    """进度日志器"""
    
    def __init__(self, total_steps: int, log_interval: int = 100):
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.current_step = 0
        
        self.logger = get_logger("ProgressLogger")
        self.start_time = datetime.now()
    
    def update(self, step: int, loss: float = None, **kwargs) -> None:
        """更新进度"""
        self.current_step = step
        
        if step % self.log_interval == 0 or step == self.total_steps:
            progress = (step / self.total_steps) * 100
            elapsed = datetime.now() - self.start_time
            
            msg = f"进度: {step}/{self.total_steps} ({progress:.1f}%), 耗时: {elapsed}"
            
            if loss is not None:
                msg += f", 损失: {loss:.4f}"
            
            for key, value in kwargs.items():
                msg += f", {key}: {value}"
            
            self.logger.info(msg)
    
    def finish(self) -> None:
        """完成进度记录"""
        total_time = datetime.now() - self.start_time
        self.logger.info(f"训练完成，总耗时: {total_time}")