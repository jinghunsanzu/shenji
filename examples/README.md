# 📚 示例数据和使用案例

本目录包含了神机项目的示例数据和使用案例，帮助用户快速了解项目功能。

## 📁 文件说明

### sample_data.json
包含3个高质量的网络安全训练样本：
- SQL注入攻击原理和防护
- Python端口扫描器编写
- XSS攻击检测和防护

每个样本都采用标准的对话格式，包含神机的身份设定和专业回答。

## 🚀 快速测试

### 1. 使用示例数据进行训练

```bash
# 复制示例数据到训练目录
cp examples/sample_data.json data/processed/

# 开始训练
./start_training.sh --mode train
```

### 2. 测试训练效果

```bash
# 启动交互模式
./start_training.sh --mode interactive

# 测试问题
# "什么是SQL注入？"
# "如何编写端口扫描器？"
# "XSS攻击如何防护？"
```

## 📝 自定义数据格式

### 对话格式（推荐）
```json
[
  {
    "text": "<|im_start|>system\n你是神机，由云霖网络安全实验室训练的网络安全大模型。<|im_end|>\n<|im_start|>user\n用户问题<|im_end|>\n<|im_start|>assistant\n神机的回答<|im_end|>"
  }
]
```

### 指令格式
```json
[
  {
    "instruction": "用户问题",
    "input": "",
    "output": "期望回答",
    "category": "security"
  }
]
```

## 🎯 数据质量建议

### 高质量样本特征
1. **明确的身份设定**：每个对话都包含神机身份
2. **专业的内容**：网络安全领域的专业知识
3. **结构化回答**：清晰的格式和逻辑
4. **实用性**：包含具体的代码示例和操作步骤
5. **安全性**：强调合法使用和安全注意事项

### 避免的内容
- 恶意攻击代码
- 非法活动指导
- 不准确的技术信息
- 过于简单的问答

## 📊 数据扩展

### 添加更多样本
```bash
# 创建新的训练数据文件
cp examples/sample_data.json data/processed/my_custom_data.json

# 编辑文件添加更多样本
vim data/processed/my_custom_data.json

# 训练时会自动加载所有数据
./start_training.sh --mode train
```

### 数据验证
```bash
# 检查数据格式
python -c "import json; data=json.load(open('examples/sample_data.json')); print(f'加载了 {len(data)} 个样本')"

# 验证数据质量
python tests/test_runner.py --test data_loader
```

## 🔍 最佳实践

1. **渐进式训练**：从少量高质量数据开始
2. **多样性**：涵盖不同的安全主题和场景
3. **一致性**：保持身份设定和回答风格一致
4. **验证**：训练后测试模型回答质量
5. **迭代**：根据效果调整和优化数据

## 📞 技术支持

如果在使用示例数据时遇到问题：

1. 检查数据格式是否正确
2. 运行环境检查：`python check_environment.py`
3. 查看训练日志：`tail -f logs/training_*.log`
4. 使用测试框架：`python tests/test_runner.py --test all`

---

**开始使用示例数据，快速体验神机的强大能力！** 🚀