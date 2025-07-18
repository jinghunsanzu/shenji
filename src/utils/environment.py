# -*- coding: utf-8 -*-
"""
环境检查工具模块

负责检查训练环境的各项配置。
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any

from .logger import get_logger


class EnvironmentChecker:
    """环境检查器"""
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.check_results = {}
    
    def check_all(self) -> Dict[str, Any]:
        """执行所有环境检查"""
        self.logger.info("开始环境检查...")
        
        checks = [
            ("python_version", self.check_python_version),
            ("cuda_availability", self.check_cuda),
            ("gpu_memory", self.check_gpu_memory),
            ("disk_space", self.check_disk_space),
            ("system_memory", self.check_system_memory),
            ("required_packages", self.check_required_packages),
            ("network_connectivity", self.check_network),
            ("directory_permissions", self.check_directory_permissions)
        ]
        
        for check_name, check_func in checks:
            try:
                result = check_func()
                self.check_results[check_name] = result
                status = "✓" if result.get("status", False) else "✗"
                self.logger.info(f"{status} {check_name}: {result.get('message', '')}")
            except Exception as e:
                self.check_results[check_name] = {
                    "status": False,
                    "message": f"检查失败: {str(e)}"
                }
                self.logger.error(f"✗ {check_name}: 检查失败 - {e}")
        
        # 生成总结
        self._generate_summary()
        
        return self.check_results
    
    def check_python_version(self) -> Dict[str, Any]:
        """检查Python版本"""
        version = sys.version_info
        version_str = f"{version.major}.{version.minor}.{version.micro}"
        
        # 要求Python 3.8+
        is_valid = version >= (3, 8)
        
        return {
            "status": is_valid,
            "version": version_str,
            "message": f"Python {version_str}" + ("" if is_valid else " (需要 3.8+)")
        }
    
    def check_cuda(self) -> Dict[str, Any]:
        """检查CUDA可用性"""
        try:
            import torch
            
            cuda_available = torch.cuda.is_available()
            
            if cuda_available:
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0)
                cuda_version = torch.version.cuda
                
                return {
                    "status": True,
                    "device_count": device_count,
                    "device_name": device_name,
                    "cuda_version": cuda_version,
                    "message": f"CUDA {cuda_version}, {device_count} GPU(s), {device_name}"
                }
            else:
                return {
                    "status": False,
                    "message": "CUDA不可用，将使用CPU训练"
                }
        except ImportError:
            return {
                "status": False,
                "message": "PyTorch未安装"
            }
    
    def check_gpu_memory(self) -> Dict[str, Any]:
        """检查GPU内存"""
        try:
            import torch
            
            if not torch.cuda.is_available():
                return {
                    "status": False,
                    "message": "无GPU可用"
                }
            
            device = torch.cuda.current_device()
            total_memory = torch.cuda.get_device_properties(device).total_memory
            total_gb = total_memory / (1024**3)
            
            # 检查可用内存
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated(device)
            cached = torch.cuda.memory_reserved(device)
            free_memory = total_memory - max(allocated, cached)
            free_gb = free_memory / (1024**3)
            
            # 至少需要4GB可用内存
            is_sufficient = free_gb >= 4.0
            
            return {
                "status": is_sufficient,
                "total_gb": round(total_gb, 2),
                "free_gb": round(free_gb, 2),
                "message": f"总内存: {total_gb:.1f}GB, 可用: {free_gb:.1f}GB" + 
                          ("" if is_sufficient else " (建议至少4GB)")
            }
        except Exception as e:
            return {
                "status": False,
                "message": f"GPU内存检查失败: {str(e)}"
            }
    
    def check_disk_space(self) -> Dict[str, Any]:
        """检查磁盘空间"""
        try:
            current_dir = Path.cwd()
            total, used, free = shutil.disk_usage(current_dir)
            
            free_gb = free / (1024**3)
            total_gb = total / (1024**3)
            
            # 至少需要10GB可用空间
            is_sufficient = free_gb >= 10.0
            
            return {
                "status": is_sufficient,
                "total_gb": round(total_gb, 2),
                "free_gb": round(free_gb, 2),
                "message": f"总空间: {total_gb:.1f}GB, 可用: {free_gb:.1f}GB" +
                          ("" if is_sufficient else " (建议至少10GB)")
            }
        except Exception as e:
            return {
                "status": False,
                "message": f"磁盘空间检查失败: {str(e)}"
            }
    
    def check_system_memory(self) -> Dict[str, Any]:
        """检查系统内存"""
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            total_gb = memory.total / (1024**3)
            available_gb = memory.available / (1024**3)
            
            # 至少需要8GB总内存
            is_sufficient = total_gb >= 8.0
            
            return {
                "status": is_sufficient,
                "total_gb": round(total_gb, 2),
                "available_gb": round(available_gb, 2),
                "usage_percent": memory.percent,
                "message": f"总内存: {total_gb:.1f}GB, 可用: {available_gb:.1f}GB" +
                          ("" if is_sufficient else " (建议至少8GB)")
            }
        except ImportError:
            return {
                "status": False,
                "message": "psutil未安装，无法检查系统内存"
            }
        except Exception as e:
            return {
                "status": False,
                "message": f"系统内存检查失败: {str(e)}"
            }
    
    def check_required_packages(self) -> Dict[str, Any]:
        """检查必需的Python包"""
        required_packages = [
            "torch", "transformers", "datasets", "peft",
            "accelerate", "bitsandbytes", "tqdm", "psutil"
        ]
        
        missing_packages = []
        installed_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                installed_packages.append(package)
            except ImportError:
                missing_packages.append(package)
        
        is_complete = len(missing_packages) == 0
        
        message = f"已安装: {len(installed_packages)}/{len(required_packages)}"
        if missing_packages:
            message += f", 缺失: {', '.join(missing_packages)}"
        
        return {
            "status": is_complete,
            "installed": installed_packages,
            "missing": missing_packages,
            "message": message
        }
    
    def check_network(self) -> Dict[str, Any]:
        """检查网络连接"""
        test_urls = [
            ("HuggingFace", "https://huggingface.co"),
            ("ModelScope", "https://modelscope.cn")
        ]
        
        connectivity_results = {}
        
        for name, url in test_urls:
            try:
                import urllib.request
                urllib.request.urlopen(url, timeout=10)
                connectivity_results[name] = True
            except Exception:
                connectivity_results[name] = False
        
        any_connected = any(connectivity_results.values())
        
        status_msg = ", ".join([
            f"{name}: {'✓' if status else '✗'}"
            for name, status in connectivity_results.items()
        ])
        
        return {
            "status": any_connected,
            "connectivity": connectivity_results,
            "message": status_msg
        }
    
    def check_directory_permissions(self) -> Dict[str, Any]:
        """检查目录权限"""
        test_dirs = [
            Path.cwd(),
            Path.cwd() / "data",
            Path.cwd() / "models",
            Path.cwd() / "logs",
            Path.cwd() / "output"
        ]
        
        permission_results = {}
        
        for dir_path in test_dirs:
            try:
                # 创建目录（如果不存在）
                dir_path.mkdir(parents=True, exist_ok=True)
                
                # 测试写入权限
                test_file = dir_path / ".permission_test"
                test_file.write_text("test")
                test_file.unlink()
                
                permission_results[str(dir_path)] = True
            except Exception:
                permission_results[str(dir_path)] = False
        
        all_writable = all(permission_results.values())
        
        failed_dirs = [
            path for path, writable in permission_results.items()
            if not writable
        ]
        
        message = "所有目录可写" if all_writable else f"无法写入: {', '.join(failed_dirs)}"
        
        return {
            "status": all_writable,
            "permissions": permission_results,
            "message": message
        }
    
    def _generate_summary(self) -> None:
        """生成检查总结"""
        total_checks = len(self.check_results)
        passed_checks = sum(1 for result in self.check_results.values() if result.get("status", False))
        
        self.logger.info(f"\n环境检查完成: {passed_checks}/{total_checks} 项通过")
        
        if passed_checks == total_checks:
            self.logger.info("✓ 环境检查全部通过，可以开始训练")
        else:
            self.logger.warning("⚠ 部分环境检查未通过，可能影响训练效果")
            
            # 列出失败的检查
            failed_checks = [
                name for name, result in self.check_results.items()
                if not result.get("status", False)
            ]
            
            self.logger.warning(f"失败的检查项: {', '.join(failed_checks)}")
    
    def get_recommendations(self) -> List[str]:
        """获取环境改进建议"""
        recommendations = []
        
        # Python版本建议
        if not self.check_results.get("python_version", {}).get("status", False):
            recommendations.append("升级Python到3.8或更高版本")
        
        # CUDA建议
        if not self.check_results.get("cuda_availability", {}).get("status", False):
            recommendations.append("安装CUDA和PyTorch GPU版本以加速训练")
        
        # 内存建议
        gpu_memory = self.check_results.get("gpu_memory", {})
        if gpu_memory.get("status") is False and "free_gb" in gpu_memory:
            if gpu_memory["free_gb"] < 4:
                recommendations.append("释放GPU内存或使用更小的batch size")
        
        # 磁盘空间建议
        disk_space = self.check_results.get("disk_space", {})
        if not disk_space.get("status", False):
            recommendations.append("清理磁盘空间，至少保留10GB可用空间")
        
        # 包安装建议
        packages = self.check_results.get("required_packages", {})
        if packages.get("missing"):
            missing = ", ".join(packages["missing"])
            recommendations.append(f"安装缺失的包: pip install {missing}")
        
        # 网络建议
        if not self.check_results.get("network_connectivity", {}).get("status", False):
            recommendations.append("检查网络连接，确保可以访问模型下载源")
        
        return recommendations