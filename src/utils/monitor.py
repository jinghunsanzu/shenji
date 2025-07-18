# -*- coding: utf-8 -*-
"""
训练监控工具模块

负责监控训练进度、系统状态和资源使用情况。
"""

import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

from .logger import get_logger


@dataclass
class SystemStatus:
    """系统状态数据类"""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_used_percent: float
    disk_free_gb: float
    gpu_memory_used_gb: float = 0.0
    gpu_memory_total_gb: float = 0.0
    gpu_utilization: float = 0.0
    gpu_temperature: float = 0.0


@dataclass
class TrainingStatus:
    """训练状态数据类"""
    timestamp: str
    epoch: int
    step: int
    loss: float
    learning_rate: float
    progress_percent: float
    estimated_time_remaining: str
    samples_per_second: float = 0.0


class TrainingMonitor:
    """训练监控器"""
    
    def __init__(self, log_dir: Optional[Path] = None, 
                 monitor_interval: int = 30):
        self.log_dir = log_dir or Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.monitor_interval = monitor_interval
        self.logger = get_logger(self.__class__.__name__)
        
        # 监控数据文件
        self.system_log_file = self.log_dir / "system_monitor.jsonl"
        self.training_log_file = self.log_dir / "training_monitor.jsonl"
        
        # 状态缓存
        self.last_system_status = None
        self.last_training_status = None
        
        # 训练开始时间
        self.training_start_time = None
        
        self.logger.info(f"训练监控器已初始化，监控间隔: {monitor_interval}秒")
    
    def start_monitoring(self) -> None:
        """开始监控"""
        self.training_start_time = datetime.now()
        self.logger.info("开始训练监控")
    
    def stop_monitoring(self) -> None:
        """停止监控"""
        if self.training_start_time:
            total_time = datetime.now() - self.training_start_time
            self.logger.info(f"训练监控结束，总时长: {total_time}")
    
    def collect_system_status(self) -> SystemStatus:
        """收集系统状态"""
        try:
            import psutil
            
            # CPU和内存信息
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            status = SystemStatus(
                timestamp=datetime.now().isoformat(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_gb=memory.used / (1024**3),
                memory_total_gb=memory.total / (1024**3),
                disk_used_percent=(disk.used / disk.total) * 100,
                disk_free_gb=disk.free / (1024**3)
            )
            
            # GPU信息（如果可用）
            try:
                import torch
                if torch.cuda.is_available():
                    device = torch.cuda.current_device()
                    gpu_memory_used = torch.cuda.memory_allocated(device)
                    gpu_memory_total = torch.cuda.get_device_properties(device).total_memory
                    
                    status.gpu_memory_used_gb = gpu_memory_used / (1024**3)
                    status.gpu_memory_total_gb = gpu_memory_total / (1024**3)
                    status.gpu_utilization = (gpu_memory_used / gpu_memory_total) * 100
                    
                    # 尝试获取GPU温度（需要nvidia-ml-py）
                    try:
                        import pynvml
                        pynvml.nvmlInit()
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        status.gpu_temperature = temp
                    except:
                        pass
            except:
                pass
            
            self.last_system_status = status
            return status
            
        except ImportError:
            self.logger.warning("psutil未安装，无法收集系统状态")
            return None
        except Exception as e:
            self.logger.error(f"收集系统状态失败: {e}")
            return None
    
    def log_training_step(self, epoch: int, step: int, loss: float, 
                         learning_rate: float, total_steps: int) -> None:
        """记录训练步骤"""
        progress_percent = (step / total_steps) * 100 if total_steps > 0 else 0
        
        # 估算剩余时间
        estimated_time = "未知"
        samples_per_second = 0.0
        
        if self.training_start_time and step > 0:
            elapsed_time = datetime.now() - self.training_start_time
            time_per_step = elapsed_time.total_seconds() / step
            remaining_steps = total_steps - step
            remaining_seconds = time_per_step * remaining_steps
            
            estimated_time = str(timedelta(seconds=int(remaining_seconds)))
            samples_per_second = step / elapsed_time.total_seconds()
        
        status = TrainingStatus(
            timestamp=datetime.now().isoformat(),
            epoch=epoch,
            step=step,
            loss=loss,
            learning_rate=learning_rate,
            progress_percent=progress_percent,
            estimated_time_remaining=estimated_time,
            samples_per_second=samples_per_second
        )
        
        self.last_training_status = status
        
        # 保存到文件
        self._save_training_status(status)
        
        # 定期收集系统状态
        if step % (self.monitor_interval // 10) == 0:  # 更频繁的系统监控
            system_status = self.collect_system_status()
            if system_status:
                self._save_system_status(system_status)
    
    def _save_system_status(self, status: SystemStatus) -> None:
        """保存系统状态到文件"""
        try:
            with open(self.system_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(status.__dict__, ensure_ascii=False) + '\n')
        except Exception as e:
            self.logger.error(f"保存系统状态失败: {e}")
    
    def _save_training_status(self, status: TrainingStatus) -> None:
        """保存训练状态到文件"""
        try:
            with open(self.training_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(status.__dict__, ensure_ascii=False) + '\n')
        except Exception as e:
            self.logger.error(f"保存训练状态失败: {e}")
    
    def get_current_status(self) -> Dict[str, Any]:
        """获取当前状态"""
        current_system = self.collect_system_status()
        
        status = {
            "monitoring_active": self.training_start_time is not None,
            "training_duration": None,
            "system_status": current_system.__dict__ if current_system else None,
            "training_status": self.last_training_status.__dict__ if self.last_training_status else None
        }
        
        if self.training_start_time:
            duration = datetime.now() - self.training_start_time
            status["training_duration"] = str(duration)
        
        return status
    
    def get_training_history(self, last_n: int = 100) -> List[Dict[str, Any]]:
        """获取训练历史"""
        if not self.training_log_file.exists():
            return []
        
        try:
            with open(self.training_log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 获取最后N行
            recent_lines = lines[-last_n:] if len(lines) > last_n else lines
            
            history = []
            for line in recent_lines:
                try:
                    data = json.loads(line.strip())
                    history.append(data)
                except json.JSONDecodeError:
                    continue
            
            return history
            
        except Exception as e:
            self.logger.error(f"读取训练历史失败: {e}")
            return []
    
    def get_system_history(self, last_n: int = 100) -> List[Dict[str, Any]]:
        """获取系统监控历史"""
        if not self.system_log_file.exists():
            return []
        
        try:
            with open(self.system_log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            recent_lines = lines[-last_n:] if len(lines) > last_n else lines
            
            history = []
            for line in recent_lines:
                try:
                    data = json.loads(line.strip())
                    history.append(data)
                except json.JSONDecodeError:
                    continue
            
            return history
            
        except Exception as e:
            self.logger.error(f"读取系统历史失败: {e}")
            return []
    
    def generate_report(self) -> Dict[str, Any]:
        """生成监控报告"""
        training_history = self.get_training_history()
        system_history = self.get_system_history()
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "training_summary": self._analyze_training_data(training_history),
            "system_summary": self._analyze_system_data(system_history),
            "current_status": self.get_current_status()
        }
        
        return report
    
    def _analyze_training_data(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析训练数据"""
        if not history:
            return {"status": "无数据"}
        
        losses = [item["loss"] for item in history if "loss" in item]
        
        if not losses:
            return {"status": "无损失数据"}
        
        return {
            "total_steps": len(history),
            "latest_loss": losses[-1],
            "min_loss": min(losses),
            "max_loss": max(losses),
            "avg_loss": sum(losses) / len(losses),
            "loss_trend": "下降" if len(losses) > 1 and losses[-1] < losses[0] else "上升",
            "latest_progress": history[-1].get("progress_percent", 0),
            "estimated_completion": history[-1].get("estimated_time_remaining", "未知")
        }
    
    def _analyze_system_data(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析系统数据"""
        if not history:
            return {"status": "无数据"}
        
        latest = history[-1]
        
        # 计算平均值
        cpu_usage = [item["cpu_percent"] for item in history if "cpu_percent" in item]
        memory_usage = [item["memory_percent"] for item in history if "memory_percent" in item]
        gpu_usage = [item["gpu_utilization"] for item in history if "gpu_utilization" in item]
        
        summary = {
            "latest_cpu_percent": latest.get("cpu_percent", 0),
            "latest_memory_percent": latest.get("memory_percent", 0),
            "latest_gpu_utilization": latest.get("gpu_utilization", 0),
            "disk_free_gb": latest.get("disk_free_gb", 0)
        }
        
        if cpu_usage:
            summary["avg_cpu_percent"] = sum(cpu_usage) / len(cpu_usage)
        
        if memory_usage:
            summary["avg_memory_percent"] = sum(memory_usage) / len(memory_usage)
        
        if gpu_usage:
            summary["avg_gpu_utilization"] = sum(gpu_usage) / len(gpu_usage)
        
        # 资源警告
        warnings = []
        if latest.get("memory_percent", 0) > 90:
            warnings.append("内存使用率过高")
        if latest.get("disk_free_gb", 0) < 2:
            warnings.append("磁盘空间不足")
        if latest.get("gpu_memory_used_gb", 0) / latest.get("gpu_memory_total_gb", 1) > 0.95:
            warnings.append("GPU内存使用率过高")
        
        summary["warnings"] = warnings
        
        return summary
    
    def save_report(self, filename: Optional[str] = None) -> Path:
        """保存监控报告"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_report_{timestamp}.json"
        
        report_path = self.log_dir / filename
        report = self.generate_report()
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"监控报告已保存: {report_path}")
        return report_path