# 安装指南

本指南将帮助您快速安装和配置项目所需的所有依赖。

## 系统要求

- Python 3.8 或更高版本
- 至少 8GB 内存
- 推荐使用 GPU（NVIDIA显卡）进行训练

## 安装步骤

### 1. 创建虚拟环境（推荐）

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 2. 安装依赖

**重要：请按照以下顺序安装，不要跳过任何步骤！**

```bash
# 方法1：一键安装所有依赖（推荐）
pip install -r requirements.txt

# 方法2：如果上述方法失败，请逐步安装
pip install torch>=2.0.0
pip install transformers>=4.30.0
pip install datasets>=2.12.0
pip install peft>=0.4.0
pip install bitsandbytes>=0.39.0
pip install accelerate>=0.20.0
pip install huggingface_hub>=0.16.0
pip install safetensors>=0.3.0
pip install sentencepiece>=0.1.99
pip install tokenizers>=0.13.0
pip install tqdm>=4.65.0
pip install requests>=2.31.0
pip install numpy>=1.21.0
pip install pandas>=1.5.0
pip install modelscope>=1.9.0
pip install psutil>=5.9.0
```

### 3. 验证安装

运行以下命令检查安装是否成功：

```bash
python -c "import torch; print('PyTorch版本:', torch.__version__)"
python -c "import transformers; print('Transformers版本:', transformers.__version__)"
python -c "import datasets; print('Datasets安装成功')"
```

### 4. 可选依赖

如果您需要实验跟踪和可视化功能，可以安装以下可选依赖：

```bash
# 实验跟踪
pip install wandb>=0.15.0

# 可视化
pip install tensorboard>=2.10.0
pip install matplotlib>=3.5.0
pip install seaborn>=0.11.0
```

## 常见问题

### Q: 安装 bitsandbytes 失败怎么办？
A: 这通常是因为缺少CUDA环境。如果您没有GPU，可以跳过这个依赖，但会影响模型量化功能。

### Q: 安装过程中出现网络错误？
A: 可以使用国内镜像源：
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

### Q: 内存不足怎么办？
A: 确保您的系统至少有8GB内存，并关闭其他占用内存的程序。

## 下一步

安装完成后，您可以：
1. 运行 `python main.py --help` 查看可用选项
2. 使用 `./start_training.sh --help` 查看训练脚本帮助
3. 阅读 README.md 了解详细使用方法

如果遇到问题，请检查错误信息并参考上述常见问题解答。