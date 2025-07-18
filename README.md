# 神机 - 网络安全大模型训练系统

神机是一个专门针对网络安全领域的AI大模型训练系统，基于Qwen2.5-1.5B模型进行LoRA微调。由云霖网络安全实验室开发，采用模块化架构设计，具备完整的身份认知和专业的网络安全知识体系。

## 🚀 功能特点

### 核心功能
- **神机身份认知**: 模型具备完整的"神机"身份认知，由云霖网络安全实验室训练
- **动态身份设置**: 支持在推理时动态设置模型身份，无需修改模型文件
- **网络安全专精**: 专门针对网络安全、渗透测试、代码审计等场景优化
- **全自动化训练**: 一键启动，自动完成数据下载、环境配置、模型训练全流程
- **统一测试框架**: 集成化测试系统，支持多种测试场景
- **小显存优化**: 针对6GB显存进行优化，支持LoRA微调
- **中文理解**: 增强中文网络安全术语和概念的理解能力
- **实时监控**: 提供训练进度监控和系统状态监控
- **模块化架构**: 采用分层架构，代码结构清晰，易于维护和扩展

### 技术特性
- 基于Qwen2.5-1.5B模型
- LoRA (Low-Rank Adaptation) 微调技术
- Chat Template动态身份设置
- 支持ModelScope和HuggingFace双源下载
- 自动混合精度训练 (FP16)
- 梯度累积和检查点保存
- 多数据源融合训练
- 统一测试框架
- 模块化设计，易于扩展

## 📋 系统要求

### 硬件要求
- **GPU**: NVIDIA P106-100 6GB 或更高
- **内存**: 16GB+ 推荐
- **存储**: 50GB+ 可用空间
- **网络**: 稳定的网络连接（用于下载模型和数据）

### 软件要求
- **操作系统**: Linux (Ubuntu 18.04+)
- **Python**: 3.8+
- **CUDA**: 11.0+
- **其他**: screen, git

## 🚀 快速开始

### ⚡ 5分钟快速上手

**新用户推荐**: 查看 [5分钟快速开始指南](QUICKSTART.md) 快速上手！

### 1. 环境准备

```bash
# 克隆或下载项目文件
cd /qwen

# 给启动脚本执行权限
chmod +x start_training.sh

# 安装系统依赖
sudo apt-get update
sudo apt-get install -y screen python3-venv python3-pip
```

### 2. 环境检查（推荐）

```bash
# 🔍 一键检查环境是否就绪
python check_environment.py

# 或使用启动脚本进行环境检查
./start_training.sh --mode check
```

如果环境检查失败，请按照提示安装缺失的依赖：
```bash
pip install -r requirements.txt
```

### 3. 一键启动训练

```bash
# 启动自动化训练（推荐）
bash start_training.sh
```

这个脚本会自动完成：
- 创建Python虚拟环境
- 安装所有依赖包
- 下载和处理训练数据
- 启动后台训练
- 设置监控和日志

### 4. 自定义训练数据（可选）

#### 放置数据文件
将你的训练数据文件放在：
- **自动加载模式**：`/qwen/data/processed/` 目录下的任意 `.json` 文件
- **指定文件模式**：`/qwen/data/processed/your_custom_name.json`

#### 程序自动加载
- **默认行为**：程序会自动扫描并加载 `/qwen/data/processed/` 目录下的所有 `.json` 文件
- **多文件支持**：可以同时放置多个训练文件，程序会自动合并所有数据
- **指定文件**：如需加载特定文件，可在训练时指定文件名

#### 数据文件格式
训练数据应为JSON格式，支持以下两种结构：

**方式1：对话格式（推荐）**
```json
[
  {
    "text": "<|im_start|>system\n你是一个网络安全专家...<|im_end|>\n<|im_start|>user\n如何进行SQL注入测试？<|im_end|>\n<|im_start|>assistant\nSQL注入测试的步骤包括...<|im_end|>"
  },
  {
    "text": "<|im_start|>system\n你是一个编程专家...<|im_end|>\n<|im_start|>user\n编写一个端口扫描器<|im_end|>\n<|im_start|>assistant\n以下是端口扫描器的代码...<|im_end|>"
  }
]
```

**方式2：指令格式**
```json
[
  {
    "instruction": "如何进行SQL注入测试？",
    "input": "",
    "output": "SQL注入测试的步骤包括...",
    "category": "security"
  }
]
```

**使用示例：**
```bash
# 创建数据目录
mkdir -p /qwen/data/processed

# 放置多个训练文件
cp my_security_data.json /qwen/data/processed/
cp my_code_data.json /qwen/data/processed/
cp my_custom_data.json /qwen/data/processed/

# 启动训练（自动加载所有文件）
./start_training.sh --mode train
```

### 3. 详细使用方法

#### 启动脚本选项
```bash
# 显示帮助
./start_training.sh --help

# 完整训练流程 (默认)
./start_training.sh
./start_training.sh --mode full

# 仅数据下载和处理
./start_training.sh --mode data

# 强制重新下载数据
./start_training.sh --mode data --force-download

# 仅模型训练
./start_training.sh --mode train

# 仅模型测试
./start_training.sh --mode test

# 交互式对话
./start_training.sh --mode interactive

# 环境检查
./start_training.sh --mode check

# 详细的依赖检查
python check_dependencies.py
```

#### 直接使用Python
```bash
# 设置Python路径
export PYTHONPATH="$PWD/src:$PYTHONPATH"

# 运行主程序
python main.py --mode full
python main.py --mode data
python main.py --mode train
python main.py --mode test
python main.py --mode interactive
```

#### 监控训练
```bash
# 查看screen会话
screen -r security_model_training

# 查看日志
tail -f logs/training_*.log

# 监控系统状态
python -c "from src.utils.monitor import TrainingMonitor; m=TrainingMonitor(); print(m.get_current_status())"
```

## 📁 项目结构

```
qwen/
├── requirements.txt          # Python依赖包
├── requirements-basic.txt    # 基础依赖包
├── main.py                  # 主程序入口
├── start_training.sh        # 启动脚本
├── download_model.py        # 模型下载脚本
├── check_environment.py     # 环境检查脚本
├── README.md               # 项目说明文档
├── QUICKSTART.md           # 5分钟快速开始指南
├── INSTALL.md              # 详细安装指南
├── identity_solution_guide.md # 身份解决方案说明
├── src/                    # 源代码目录
│   ├── __init__.py         # 包初始化
│   ├── app.py              # 主应用程序
│   ├── config/             # 配置模块
│   │   ├── __init__.py
│   │   ├── settings.py     # 基础配置
│   │   ├── training_config.py  # 训练配置
│   │   └── data_config.py  # 数据配置（含神机身份模板）
│   ├── data/               # 数据处理模块
│   │   ├── __init__.py
│   │   ├── downloader.py   # 数据下载器
│   │   ├── processor.py    # 数据处理器
│   │   └── loader.py       # 数据加载器
│   ├── model/              # 模型管理模块
│   │   ├── __init__.py
│   │   ├── downloader.py   # 模型下载器
│   │   ├── trainer.py      # 模型训练器
│   │   └── inference.py    # 模型推理器（支持动态身份）
│   └── utils/              # 工具模块
│       ├── __init__.py
│       ├── logger.py       # 日志工具
│       ├── environment.py  # 环境检查
│       └── monitor.py      # 训练监控
├── tests/                  # 统一测试框架
│   ├── __init__.py         # 测试模块初始化
│   ├── test_runner.py      # 统一测试运行器
│   └── README.md          # 测试框架说明
├── examples/               # 示例数据和使用案例
│   ├── sample_data.json    # 示例训练数据
│   └── README.md          # 示例使用说明
├── data/                   # 数据目录
│   └── processed/         # 处理后数据
│       ├── final_security_training_dataset.jsonl
│       ├── security_only_training_dataset.jsonl
│       ├── enhanced_test.jsonl
│       └── ...            # 其他训练数据
├── models/                # 模型目录
│   └── Qwen_Qwen2.5-1.5B-Instruct/  # 基础模型
│       ├── config.json
│       ├── tokenizer_config.json  # 包含神机身份模板
│       └── ...            # 其他模型文件
├── checkpoints/           # 训练检查点
│   └── checkpoint-6/      # LoRA适配器
│       ├── adapter_model.safetensors
│       ├── chat_template.jinja  # 神机身份模板
│       └── ...            # 其他检查点文件
├── logs/                  # 日志目录
├── cache/                 # 缓存目录
├── output/                # 输出目录
└── venv/                  # Python虚拟环境
```

## 🔧 配置说明

### 配置文件结构

配置采用模块化设计，分为三个主要部分：

#### 1. 基础配置 (src/config/settings.py)
```python
class Config:
    # 项目路径配置
    PROJECT_ROOT = "/path/to/project"
    DATA_DIR = "data"
    MODEL_DIR = "models"
    
    # 模型配置
    BASE_MODEL_NAME = "Qwen/Qwen2-1.5B"
    USE_MODELSCOPE = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

#### 2. 训练配置 (src/config/training_config.py)
```python
class TrainingConfig:
    # 训练参数
    batch_size = 2
    learning_rate = 2e-4
    num_epochs = 3
    max_length = 2048
    
class LoRAConfig:
    # LoRA参数
    r = 16
    alpha = 32
    dropout = 0.1
```

#### 3. 数据配置 (src/config/data_config.py)
```python
class DataConfig:
    # 数据处理参数
    max_samples_per_source = 1000
    train_test_split = 0.8
    min_length = 10
    max_length = 2048
```

### 数据源配置

系统会自动从以下来源获取训练数据：
- 网络安全知识库
- 代码数据集
- 中文对话数据
- WebShell示例代码
- 渗透测试脚本

## 🔧 高级用法

### 自定义配置

1. **修改训练参数**
```python
# 编辑 src/config/training_config.py
class TrainingConfig:
    batch_size = 4          # 增加批次大小
    learning_rate = 1e-4    # 降低学习率
    num_epochs = 5          # 增加训练轮数
```

2. **添加自定义数据源**
```python
# 在 src/data/downloader.py 中添加数据源
class DataDownloader:
    def download_custom_data(self):
        custom_data = [
            {
                "instruction": "你的指令",
                "input": "输入内容",
                "output": "期望输出"
            }
        ]
        return custom_data
```

3. **自定义提示模板**
```python
# 在 src/config/data_config.py 中修改模板
class PromptTemplates:
    SECURITY_ANALYSIS = "分析以下安全问题：{question}\n\n分析：{answer}"
```

### 模块化使用

1. **单独使用数据处理模块**
```python
from src.data import DataDownloader, DataProcessor

downloader = DataDownloader()
processor = DataProcessor()

# 下载和处理数据
data = downloader.download_all_data()
processed_data = processor.process_data(data)
```

2. **单独使用模型训练模块**
```python
from src.model import SecurityModelTrainer
from src.config import TrainingConfig, LoRAConfig

trainer = SecurityModelTrainer()
trainer.setup_model_and_tokenizer()
trainer.train()
```

3.### 单独使用推理模块**
```python
from src.model import SecurityModelInference
from src.config.settings import Config

# 使用默认模型
config = Config()
inference = SecurityModelInference(config)
inference.load_model()

# 使用指定模型
inference = SecurityModelInference(config, model_key="qwen2.5-7b")
inference.load_model(model_key="qwen2.5-7b")

response = inference.generate_response("如何进行SQL注入测试？")
```

### 模型管理

使用模型管理CLI工具来管理多个模型：

```bash
# 列出所有可用模型
python scripts/model_manager.py list

# 下载指定模型
python scripts/model_manager.py download qwen2.5-1.5b
python scripts/model_manager.py download chatglm3-6b

# 切换当前模型
python scripts/model_manager.py switch qwen2.5-7b

# 显示当前模型信息
python scripts/model_manager.py current

# 测试模型推理
python scripts/model_manager.py test qwen2.5-1.5b

# 与模型进行交互式对话
python scripts/model_manager.py chat qwen2.5-1.5b

# 显示模型详细信息
python scripts/model_manager.py info chatglm3-6b
``` 训练管理

```bash
# 查看所有screen会话
screen -list

# 进入训练会话
screen -r security_training

# 在会话内分离（不终止训练）
# 按 Ctrl+A, 然后按 D

# 终止训练
screen -S security_training -X quit

# 查看GPU使用情况
watch -n 1 nvidia-smi
```

## 📊 训练监控

### 监控界面

运行 `python3 monitor.py` 可以看到：
- 训练会话状态
- 当前训练进度
- GPU使用情况
- 系统资源状态
- 最新训练日志

### 日志文件

- `logs/training_YYYYMMDD_HHMMSS.log`: 训练日志
- `training.log`: 主程序日志
- `training_report.txt`: 训练完成报告

## 🤖 神机身份解决方案

### 身份认知特性
神机具备完整的身份认知系统，确保模型始终以"神机"身份回答问题：

- **默认身份**: "你是神机，由云霖网络安全实验室训练的网络安全大模型"
- **动态设置**: 支持在推理时动态调整身份，无需修改模型文件
- **Chat Template**: 使用标准化的对话模板确保身份一致性
- **多轮对话**: 在整个对话过程中保持身份认知

### 技术实现
```python
# 使用默认神机身份
from src.model.inference import SecurityModelInference
inference = SecurityModelInference()
response, history = inference.chat("你是谁？")

# 使用自定义身份
custom_system = "你是一个专业的网络安全专家。"
response, history = inference.chat("你是谁？", system_message=custom_system)
```

详细说明请参考：[身份解决方案指南](identity_solution_guide.md)

## 🧪 统一测试框架

### 测试功能
项目集成了统一的测试框架，支持以下测试项目：

- **identity**: 身份解决方案测试
- **data_loader**: 数据加载器测试
- **download**: 模型下载功能测试
- **git_download**: Git下载功能测试
- **inference**: 模型推理功能测试
- **all**: 运行所有测试

### 使用方法
```bash
# 查看所有可用测试
python tests/test_runner.py --list

# 运行单项测试
python tests/test_runner.py --test identity

# 运行所有测试
python tests/test_runner.py --test all
```

详细说明请参考：[测试框架说明](tests/README.md)

## 🎯 模型能力

神机支持多种大语言模型，每个模型都具备以下核心能力：

### 网络安全专业能力
- WebShell代码生成和分析
- 渗透测试脚本编写
- 漏洞分析和利用技术
- 安全工具开发
- 网络攻防技术指导
- CVE漏洞分析
- 安全加固建议

### 编程开发能力
- Python/PHP/Java等多语言编程
- 网络编程和系统编程
- 安全工具和脚本开发
- 代码审计和漏洞挖掘
- 自动化工具开发

### 中文理解能力
- 中文技术文档理解
- 中文安全知识问答
- 中文代码注释生成
- 中文安全术语解释

### 身份认知能力
- 明确的"神机"身份认知
- 云霖网络安全实验室背景
- 专业的网络安全知识体系
- 一致的身份表达

## 🤖 支持的模型

神机支持多种主流大语言模型，用户可以根据需求选择合适的模型：

### Qwen系列
- **Qwen2.5-1.5B-Instruct**: 轻量级模型，适合资源受限环境
- **Qwen2.5-7B-Instruct**: 平衡性能和资源消耗的中等规模模型
- **Qwen2.5-14B-Instruct**: 高性能模型，适合专业应用
- **Qwen2.5-32B-Instruct**: 大规模模型，提供最佳性能
- **Qwen2.5-72B-Instruct**: 超大规模模型，适合高端应用

### ChatGLM系列
- **ChatGLM3-6B**: 支持多轮对话的中文优化模型

### Baichuan系列
- **Baichuan2-7B-Chat**: 中文能力强的对话模型
- **Baichuan2-13B-Chat**: 更大规模的中文对话模型

### Llama系列
- **Llama2-7B-Chat**: Meta开源的对话模型
- **Llama2-13B-Chat**: 更大规模的Llama对话模型

每个模型都经过专门的适配和优化，确保在网络安全领域的专业表现。

## ⚠️ 注意事项

### 训练时间
- 预计训练时间：6-12小时
- 具体时间取决于数据量和硬件性能
- 建议在夜间或空闲时间进行训练

### 显存优化
- 如果显存不足，可以减小 `batch_size` 或 `max_length`
- 增加 `gradient_accumulation_steps` 来保持有效批次大小
- 确保启用 `fp16` 混合精度训练

### 数据安全
- 训练数据包含网络安全相关内容
- 生成的模型可能输出敏感代码
- 请确保在安全环境中使用
- 遵守相关法律法规

### 模型使用
- 神机模型专为教育和研究目的设计
- 请负责任地使用生成的安全工具代码
- 不要用于非法攻击活动
- 模型具备明确的身份认知，请尊重其专业性

## 🔍 故障排除

### 常见错误

1. **CUDA out of memory**
```bash
# 解决方案：降低批次大小
# 编辑 src/config/training_config.py
class TrainingConfig:
    batch_size = 1
    gradient_accumulation_steps = 16
```

2. **模型下载失败**
```bash
# 解决方案：切换数据源
# 编辑 src/config/settings.py
class Config:
    USE_MODELSCOPE = False  # 切换到HuggingFace
```

3. **模块导入错误**
```bash
# 解决方案：设置Python路径
export PYTHONPATH="$PWD/src:$PYTHONPATH"

# 或使用启动脚本（自动设置）
./start_training.sh
```

4. **权限错误**
```bash
# 解决方案：修改权限
chmod +x start_training.sh
sudo chown -R $USER:$USER ./
```

### 日志分析

查看详细错误信息：
```bash
# 查看最新日志
tail -f logs/training_$(ls logs/ | grep training | tail -1)

# 搜索错误
grep -i error logs/training_*.log

# 查看GPU使用情况
watch -n 1 nvidia-smi

# 查看训练状态
python -c "from src.utils.monitor import TrainingMonitor; m=TrainingMonitor(); print(m.get_current_status())"
```

### 开发调试

```bash
# 使用统一测试框架进行调试
python tests/test_runner.py --test all

# 测试特定功能
python tests/test_runner.py --test identity      # 测试身份解决方案
python tests/test_runner.py --test data_loader   # 测试数据加载
python tests/test_runner.py --test inference     # 测试推理功能

# 测试单个模块
python -c "from src.config import Config; print(Config.PROJECT_ROOT)"
python -c "from src.data import DataDownloader; d=DataDownloader(); print('DataDownloader OK')"
python -c "from src.model import SecurityModelTrainer; print('Trainer OK')"

# 环境检查
python main.py --mode check
```

### 代码结构说明
- **src/config/**: 配置管理，包含所有配置类
- **src/data/**: 数据处理，包含下载、处理、加载功能
- **src/model/**: 模型管理，包含下载、训练、推理功能
- **src/utils/**: 工具模块，包含日志、监控、环境检查
- **src/app.py**: 主应用程序，整合所有功能

### 性能优化

1. **提高训练速度**
   - 使用SSD存储
   - 增加系统内存
   - 优化数据加载

2. **减少显存使用**
   - 启用梯度检查点
   - 使用更小的模型
   - 减少序列长度

## 📞 技术支持

如果遇到问题，请按以下步骤排查：

### 基础检查
1. 系统要求是否满足
2. 依赖是否正确安装
3. 日志文件中的错误信息
4. GPU驱动和CUDA版本

### 使用测试框架诊断
```bash
# 运行完整测试诊断
python tests/test_runner.py --test all

# 针对性测试
python tests/test_runner.py --test identity      # 身份问题
python tests/test_runner.py --test download      # 下载问题
python tests/test_runner.py --test inference     # 推理问题
```

### 常见问题
- **身份认知问题**: 参考 [身份解决方案指南](identity_solution_guide.md)
- **测试相关问题**: 参考 [测试框架说明](tests/README.md)
- **模型推理问题**: 检查Chat Template配置
- **训练问题**: 查看训练日志和监控信息

## 📄 许可证

本项目由云霖网络安全实验室开发，仅供学习和研究使用，请遵守相关法律法规。相关专利已经申请

## 🔗 相关文档

- [身份解决方案指南](identity_solution_guide.md) - 详细的神机身份设置说明
- [测试框架说明](tests/README.md) - 统一测试框架使用指南
- [项目更新日志](CHANGELOG.md) - 版本更新记录（如有）

## 🏷️ 版本信息

- **当前版本**: v2.0.0
- **当前模型版本**: 基于Qwen2.5-1.5B
- **身份系统**: 神机身份认知v1.0
- **测试框架**: 统一测试系统v1.0

---

**开始你的网络安全模型训练之旅！** 🚀🤖