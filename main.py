#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
网络安全模型训练系统 - 主入口

这是重构后的主入口文件，使用模块化架构。
"""

import sys
from pathlib import Path

# 添加src目录到Python路径
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.app import main



if __name__ == "__main__":
    main()