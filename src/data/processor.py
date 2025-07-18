# -*- coding: utf-8 -*-
"""
数据处理器模块

负责数据的清洗、格式化和预处理。
"""

import json
from typing import List, Dict, Any
from pathlib import Path
from tqdm import tqdm

from ..config import Config, DataConfig
from ..config.data_config import PromptTemplates
from ..utils.logger import get_logger


class DataProcessor:
    """数据处理器"""
    
    def __init__(self, config: Config = None, data_config: DataConfig = None):
        self.config = config or Config()
        self.data_config = data_config or DataConfig()
        self.logger = get_logger(self.__class__.__name__)
        
        # 确保处理后数据目录存在
        self.processed_data_dir = self.config.DATA_DIR / "processed"
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    def clean_data(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """清洗数据"""
        self.logger.info("开始清洗数据...")
        
        cleaned_data = []
        
        for item in tqdm(raw_data, desc="清洗数据"):
            # 基本字段检查
            if not self._is_valid_item(item):
                continue
            
            # 长度过滤
            if not self._check_length(item):
                continue
            
            # 格式化数据
            cleaned_item = self._format_item(item)
            cleaned_data.append(cleaned_item)
        
        self.logger.info(f"数据清洗完成，保留 {len(cleaned_data)}/{len(raw_data)} 条数据")
        return cleaned_data
    
    def format_for_training(self, data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """格式化数据用于训练"""
        self.logger.info("格式化训练数据...")
        
        formatted_data = []
        
        for item in tqdm(data, desc="格式化数据"):
            # 根据类别选择模板
            category = item.get("category", "general")
            template = PromptTemplates.get_template(category)
            
            # 构建训练文本
            text = template.format_conversation(
                instruction=item["instruction"],
                input_text=item.get("input", ""),
                output=item["output"]
            )
            
            formatted_data.append({"text": text})
        
        self.logger.info(f"数据格式化完成，共 {len(formatted_data)} 条训练数据")
        return formatted_data
    
    def save_processed_data(self, data: List[Dict[str, Any]], filename: str) -> Path:
        """保存处理后的数据"""
        filepath = self.processed_data_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"数据已保存到: {filepath}")
        return filepath
    
    def load_processed_data(self, filename: str) -> List[Dict[str, Any]]:
        """加载处理后的数据"""
        filepath = self.processed_data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"数据文件不存在: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.logger.info(f"从 {filepath} 加载了 {len(data)} 条数据")
        return data
    
    def process_and_save(self, raw_data: List[Dict[str, Any]], 
                        clean_filename: str = "cleaned_data.json",
                        training_filename: str = "training_data.json") -> tuple[Path, Path]:
        """处理并保存数据"""
        # 清洗数据
        cleaned_data = self.clean_data(raw_data)
        clean_path = self.save_processed_data(cleaned_data, clean_filename)
        
        # 格式化训练数据
        training_data = self.format_for_training(cleaned_data)
        training_path = self.save_processed_data(training_data, training_filename)
        
        return clean_path, training_path
    
    def _is_valid_item(self, item: Dict[str, Any]) -> bool:
        """检查数据项是否有效"""
        required_fields = ["instruction", "output"]
        
        for field in required_fields:
            if field not in item or not item[field]:
                return False
        
        return True
    
    def _check_length(self, item: Dict[str, Any]) -> bool:
        """检查数据长度"""
        instruction_len = len(item["instruction"].strip())
        output_len = len(item["output"].strip())
        
        if instruction_len < self.data_config.min_instruction_length:
            return False
        
        if output_len < self.data_config.min_output_length:
            return False
        
        return True
    
    def _format_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """格式化数据项"""
        return {
            "instruction": item["instruction"].strip(),
            "input": item.get("input", "").strip(),
            "output": item["output"].strip(),
            "category": item.get("category", "general")
        }
    
    def get_data_statistics(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """获取数据统计信息"""
        if not data:
            return {"total": 0}
        
        # 按类别统计
        category_counts = {}
        total_instruction_length = 0
        total_output_length = 0
        
        for item in data:
            category = item.get("category", "unknown")
            category_counts[category] = category_counts.get(category, 0) + 1
            
            total_instruction_length += len(item.get("instruction", ""))
            total_output_length += len(item.get("output", ""))
        
        stats = {
            "total": len(data),
            "categories": category_counts,
            "avg_instruction_length": total_instruction_length / len(data),
            "avg_output_length": total_output_length / len(data)
        }
        
        return stats