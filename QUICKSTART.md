# 🚀 5分钟快速开始

本指南将帮助您在5分钟内快速上手Qwen安全模型训练项目。

## 第一步：环境检查 (1分钟)

```bash
# 检查环境是否就绪
python check_environment.py
```

如果检查失败，请按照提示安装缺失的依赖：
```bash
pip install -r requirements.txt
```

## 第二步：查看可用选项 (1分钟)

```bash
# 查看所有可用的命令选项
./start_training.sh --help

# 或者
python main.py --help
```

## 第三步：选择运行模式 (3分钟)

### 🔍 模式1：环境检查
```bash
./start_training.sh --mode check
```

### 📊 模式2：数据准备
```bash
./start_training.sh --mode data
```

### 🤖 模式3：模型训练
```bash
# 使用默认模型训练
./start_training.sh --mode train

# 选择特定模型训练
./start_training.sh --mode train --model qwen2-1.5b

# 继续之前的训练
./start_training.sh --mode train --resume
```

### 🧪 模式4：模型测试
```bash
./start_training.sh --mode test
```

### 💬 模式5：交互模式
```bash
./start_training.sh --mode interactive
```

### 🔄 模式6：完整流程
```bash
# 一键运行：数据准备 → 训练 → 测试
./start_training.sh --mode full
```

## 常用命令组合

### 新手推荐：完整流程
```bash
# 第一次使用，运行完整流程
./start_training.sh --mode full
```

### 进阶用户：自定义训练
```bash
# 查看支持的模型
./start_training.sh --list-models

# 使用特定模型训练
./start_training.sh --mode train --model qwen2-7b

# 从检查点继续训练
./start_training.sh --mode train --resume-from ./checkpoints/checkpoint-1000
```

## 📁 重要目录说明

- `data/processed/` - 处理后的训练数据
- `models/` - 下载的预训练模型
- `checkpoints/` - 训练检查点
- `output/` - 训练输出和微调后的模型
- `logs/` - 训练日志

## 🆘 遇到问题？

1. **环境问题**：运行 `python check_environment.py`
2. **依赖问题**：查看 [INSTALL.md](INSTALL.md)
3. **使用问题**：查看 [README.md](README.md)
4. **错误信息**：检查 `logs/` 目录下的日志文件

## 🎯 下一步

- 阅读 [README.md](README.md) 了解详细功能
- 查看 [examples/](examples/) 目录的示例代码
- 自定义配置文件进行高级训练

---

**提示**：首次运行建议使用 `--mode full` 体验完整流程！