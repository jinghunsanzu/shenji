# 交互式模型选择功能

## 功能概述

现在系统支持交互式模型选择功能，在模型下载和训练时会提示用户选择要使用的模型。如果用户不输入任何内容（直接回车），系统将使用默认的 Qwen2.5-1.5B-Instruct 模型。

## 支持的模型

系统目前支持以下模型：

1. **qwen2.5-1.5b-instruct**: Qwen2.5-1.5B-Instruct (qwen2) - 默认模型
2. **qwen2-1.5b-instruct**: Qwen2-1.5B-Instruct (qwen2)
3. **qwen2-7b-instruct**: Qwen2-7B-Instruct (qwen2)
4. **chatglm3-6b**: ChatGLM3-6B (chatglm)
5. **baichuan2-7b-chat**: Baichuan2-7B-Chat (baichuan)
6. **llama2-7b-chat**: Llama2-7B-Chat (llama)

## 使用方法

### 1. 下载模型时的交互选择

```bash
python download_model.py
```

系统会显示可用模型列表，您可以：
- 输入数字编号（1-6）选择对应模型
- 直接输入模型键名（如 `qwen2-7b-instruct`）
- 直接回车使用默认模型（Qwen2.5-1.5B-Instruct）

### 2. 训练时的交互选择

```bash
# 不指定模型参数时，会提示选择
python main.py --mode train

# 或者直接指定模型
python main.py --mode train --model qwen2-7b-instruct
```

### 3. 其他模式的交互选择

```bash
# 环境检查时选择模型
python main.py --mode check

# 数据准备时选择模型
python main.py --mode data

# 完整流程时选择模型
python main.py --mode full
```

## 交互界面示例

```
=== 模型选择 ===
可用的模型:
  1. qwen2.5-1.5b-instruct: Qwen2.5-1.5B-Instruct (qwen2)
  2. qwen2-1.5b-instruct: Qwen2-1.5B-Instruct (qwen2)
  3. qwen2-7b-instruct: Qwen2-7B-Instruct (qwen2)
  4. chatglm3-6b: ChatGLM3-6B (chatglm)
  5. baichuan2-7b-chat: Baichuan2-7B-Chat (baichuan)
  6. llama2-7b-chat: Llama2-7B-Chat (llama)

默认模型: Qwen2.5-1.5B-Instruct (qwen2.5-1.5b-instruct)
请选择模型 (输入数字编号，或直接回车使用默认模型):
> 
```

## 选择方式

### 1. 数字选择
输入 `1` 到 `6` 的数字选择对应模型：
```
> 2
已选择: qwen2-1.5b-instruct
```

### 2. 模型键名选择
直接输入模型的键名：
```
> qwen2-7b-instruct
已选择: qwen2-7b-instruct
```

### 3. 默认选择
直接回车使用默认模型：
```
> 
选择的模型: qwen2.5-1.5b-instruct
```

### 4. 错误处理
如果输入无效，系统会自动使用默认模型：
```
> invalid_model
未找到模型 'invalid_model'，使用默认模型
```

## 模型特点

### Qwen 系列
- **qwen2.5-1.5b-instruct**: 最新版本，推荐使用，支持32K上下文
- **qwen2-1.5b-instruct**: 稳定版本，支持32K上下文
- **qwen2-7b-instruct**: 更大模型，性能更好但需要更多资源

### 其他模型
- **chatglm3-6b**: 清华大学开发，中文表现优秀
- **baichuan2-7b-chat**: 百川智能开发，商用友好
- **llama2-7b-chat**: Meta开发，开源社区广泛使用

## 注意事项

1. **资源需求**: 不同模型对GPU内存和计算资源的需求不同
   - 1.5B模型：约需要4-6GB GPU内存
   - 6-7B模型：约需要14-16GB GPU内存

2. **下载时间**: 模型大小不同，下载时间也不同
   - 1.5B模型：约3-4GB
   - 6-7B模型：约12-14GB

3. **训练时间**: 模型越大，训练时间越长

4. **兼容性**: 所有模型都支持LoRA微调和量化

## 后台运行建议

由于模型训练通常需要较长时间，建议使用 `screen` 或 `nohup` 命令在后台运行：

### 使用 screen（推荐）

```bash
# 安装 screen（如果未安装）
sudo apt-get install screen  # Ubuntu/Debian
sudo yum install screen      # CentOS/RHEL

# 创建新的 screen 会话并运行训练
screen -S model_training
# 在 screen 会话中运行
./start_training.sh --mode full

# 分离会话（保持训练继续）：按 Ctrl+A，然后按 D
# 重新连接会话
screen -r model_training

# 查看所有会话
screen -ls

# 终止会话
screen -S model_training -X quit
```

### 使用 nohup

```bash
# 后台运行并保存日志
nohup ./start_training.sh --mode full > training.log 2>&1 &

# 查看进程
jobs
ps aux | grep python

# 查看日志
tail -f training.log
```

## 自动化使用

如果需要在脚本中自动化使用，可以通过管道输入选择：

```bash
# 选择第2个模型
echo '2' | python download_model.py

# 使用默认模型
echo '' | python main.py --mode train

# 直接指定模型（推荐）
python main.py --mode train --model qwen2-7b-instruct

# 结合 screen 使用
screen -dmS training bash -c "echo '1' | ./start_training.sh --mode full"
```

## 故障排除

如果遇到问题：

1. **模型列表不显示**: 检查 `src/config/model_configs.py` 文件
2. **选择无效**: 确保输入的是有效的数字或模型键名
3. **下载失败**: 检查网络连接和存储空间
4. **训练失败**: 确保选择的模型与您的硬件资源匹配

## 测试功能

可以使用测试脚本验证功能：

```bash
# 测试交互式选择功能
python test_model_selection.py

# 自动化测试
echo '1' | python test_model_selection.py
```