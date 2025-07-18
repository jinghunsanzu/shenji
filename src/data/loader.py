# -*- coding: utf-8 -*-
"""
数据加载器模块

负责为模型训练准备数据集。
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from ..config import Config, DataConfig
from ..utils.logger import get_logger


class SecurityDataset(Dataset):
    """网络安全数据集"""
    
    def __init__(self, data: List[Dict[str, str]], tokenizer: PreTrainedTokenizer, 
                 max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 处理不同的数据格式
        if "text" in item:
            text = item["text"]
        elif "instruction" in item and "output" in item:
            # 格式化instruction-output格式的数据
            instruction = item["instruction"]
            input_text = item.get("input", "")
            output = item["output"]
            
            if input_text:
                text = f"<|im_start|>user\n{instruction}\n{input_text}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
            else:
                text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
        else:
            # 如果都没有，尝试将整个item转换为字符串
            text = str(item)
        
        # 分词
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": encoding["input_ids"].flatten()
        }


class DataLoader:
    """数据加载器"""
    
    def __init__(self, config: Config = None, data_config: DataConfig = None):
        self.config = config or Config()
        self.data_config = data_config or DataConfig()
        self.logger = get_logger(self.__class__.__name__)
        self.loaded_files = []  # 记录已加载的文件
    
    def load_all_json_files_from_directory(self, directory_path: Path) -> List[Dict[str, str]]:
        """从目录加载所有JSON和JSONL文件"""
        all_data = []
        # 同时查找JSON和JSONL文件
        json_files = list(directory_path.glob("*.json"))
        jsonl_files = list(directory_path.glob("*.jsonl"))
        all_files = json_files + jsonl_files
        self.loaded_files = []  # 重置已加载文件列表
        
        if not all_files:
            self.logger.warning(f"目录 {directory_path} 中未找到JSON或JSONL文件")
            return all_data
        
        self.logger.info(f"发现 {len(json_files)} 个JSON文件和 {len(jsonl_files)} 个JSONL文件")
        
        # 处理JSON文件
        for json_file in json_files:
            try:
                self.logger.info(f"正在加载JSON文件: {json_file.name}")
                with open(json_file, 'r', encoding='utf-8') as f:
                    file_data = json.load(f)
                
                # 确保数据是列表格式
                if isinstance(file_data, list):
                    all_data.extend(file_data)
                    self.logger.info(f"从 {json_file.name} 加载了 {len(file_data)} 条数据")
                    self.loaded_files.append(f"{json_file.name} ({len(file_data)}条)")
                elif isinstance(file_data, dict):
                    all_data.append(file_data)
                    self.logger.info(f"从 {json_file.name} 加载了 1 条数据")
                    self.loaded_files.append(f"{json_file.name} (1条)")
                else:
                    self.logger.warning(f"跳过文件 {json_file.name}：不支持的数据格式")
                    
            except Exception as e:
                self.logger.error(f"加载JSON文件 {json_file.name} 失败: {e}")
                continue
        
        # 处理JSONL文件
        for jsonl_file in jsonl_files:
            try:
                self.logger.info(f"正在加载JSONL文件: {jsonl_file.name}")
                file_data = []
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:  # 跳过空行
                            try:
                                json_obj = json.loads(line)
                                file_data.append(json_obj)
                            except json.JSONDecodeError as e:
                                self.logger.warning(f"跳过 {jsonl_file.name} 第 {line_num} 行：JSON解析错误 - {e}")
                                continue
                
                if file_data:
                    all_data.extend(file_data)
                    self.logger.info(f"从 {jsonl_file.name} 加载了 {len(file_data)} 条数据")
                    self.loaded_files.append(f"{jsonl_file.name} ({len(file_data)}条)")
                else:
                    self.logger.warning(f"文件 {jsonl_file.name} 中没有有效数据")
                    
            except Exception as e:
                self.logger.error(f"加载JSONL文件 {jsonl_file.name} 失败: {e}")
                continue
        
        self.logger.info(f"总共加载了 {len(all_data)} 条训练数据")
        if self.loaded_files:
            self.logger.info(f"已加载的文件: {', '.join(self.loaded_files)}")
        return all_data
    
    def load_training_data(self, data_file: Optional[str] = None) -> List[Dict[str, str]]:
        """加载训练数据
        
        Args:
            data_file: 指定文件名，如果为None则自动加载目录下所有JSON文件
        """
        processed_dir = self.config.DATA_DIR / "processed"
        
        # 确保目录存在
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        if data_file is None:
            # 自动加载目录下所有JSON文件
            self.logger.info("未指定数据文件，将自动加载目录下所有JSON文件")
            data = self.load_all_json_files_from_directory(processed_dir)
            
            if not data:
                # 如果没有找到任何文件，尝试加载默认文件
                default_file = processed_dir / "training_data.json"
                if default_file.exists():
                    self.logger.info("未找到其他文件，尝试加载默认文件 training_data.json")
                    with open(default_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                else:
                    raise FileNotFoundError(f"目录 {processed_dir} 中未找到任何训练数据文件")
        else:
            # 加载指定文件
            data_path = processed_dir / data_file
            
            if not data_path.exists():
                raise FileNotFoundError(f"训练数据文件不存在: {data_path}")
            
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.logger.info(f"加载训练数据: {len(data)} 条")
        
        return data
    
    def create_dataset(self, data: List[Dict[str, str]], 
                      tokenizer: PreTrainedTokenizer) -> SecurityDataset:
        """创建数据集"""
        dataset = SecurityDataset(
            data=data,
            tokenizer=tokenizer,
            max_length=self.data_config.max_sequence_length
        )
        
        self.logger.info(f"创建数据集: {len(dataset)} 个样本")
        return dataset
    
    def split_data(self, data: List[Dict[str, str]], 
                  train_ratio: float = 0.9) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """分割训练和验证数据"""
        total_size = len(data)
        train_size = int(total_size * train_ratio)
        
        train_data = data[:train_size]
        val_data = data[train_size:]
        
        self.logger.info(f"数据分割: 训练集 {len(train_data)} 条，验证集 {len(val_data)} 条")
        return train_data, val_data
    
    def prepare_datasets(self, tokenizer: PreTrainedTokenizer, 
                       data_file: Optional[str] = None,
                       train_ratio: float = 0.9) -> tuple[SecurityDataset, SecurityDataset]:
        """准备训练和验证数据集"""
        # 加载数据
        data = self.load_training_data(data_file)
        
        # 分割数据
        train_data, val_data = self.split_data(data, train_ratio)
        
        # 创建数据集
        train_dataset = self.create_dataset(train_data, tokenizer)
        val_dataset = self.create_dataset(val_data, tokenizer)
        
        return train_dataset, val_dataset
    
    def get_data_info(self, data_file: Optional[str] = None) -> Dict[str, Any]:
        """获取数据信息"""
        try:
            data = self.load_training_data(data_file)
            
            # 计算统计信息
            total_chars = 0
            for item in data:
                if isinstance(item, dict):
                    if "text" in item:
                        total_chars += len(item["text"])
                    elif "instruction" in item and "output" in item:
                        total_chars += len(item["instruction"]) + len(item.get("input", "")) + len(item["output"])
                    else:
                        total_chars += len(str(item))
                else:
                    total_chars += len(str(item))
            
            avg_length = total_chars / len(data) if data else 0
            
            return {
                "total_samples": len(data),
                "total_characters": total_chars,
                "average_length": avg_length,
                "max_sequence_length": self.data_config.max_sequence_length
            }
        except Exception as e:
            self.logger.error(f"获取数据信息失败: {e}")
            return {}