# 神机项目测试框架

## 概述

这是神机项目的统一测试框架，将所有测试功能整合到一个脚本中，避免项目中散落多个测试文件。

## 文件结构

```
tests/
├── __init__.py          # 测试模块初始化
├── test_runner.py       # 统一测试运行器
└── README.md           # 本说明文件
```

## 使用方法

### 基本命令

```bash
# 查看帮助
python tests/test_runner.py --help

# 列出所有可用测试
python tests/test_runner.py --list

# 运行所有测试
python tests/test_runner.py --test all
```

### 单项测试

```bash
# 测试身份解决方案
python tests/test_runner.py --test identity

# 测试数据加载器
python tests/test_runner.py --test data_loader

# 测试模型下载功能
python tests/test_runner.py --test download

# 测试Git下载功能
python tests/test_runner.py --test git_download

# 测试模型推理功能
python tests/test_runner.py --test inference
```

## 可用测试项目

### 1. identity - 身份解决方案测试
- 测试Chat Template功能
- 验证神机身份设置
- 检查推理代码集成

### 2. data_loader - 数据加载器测试
- 测试数据加载器初始化
- 检查训练数据文件
- 验证数据处理功能

### 3. download - 模型下载功能测试
- 测试模型下载器
- 检查模型文件完整性
- 验证模型配置文件

### 4. git_download - Git下载功能测试
- 检查Git环境
- 验证仓库状态
- 测试版本控制功能

### 5. inference - 模型推理功能测试
- 测试推理器初始化
- 检查模型加载
- 验证推理接口

### 6. all - 运行所有测试
- 依次执行所有单项测试
- 提供测试结果汇总
- 显示通过率统计

## 添加新测试

要添加新的测试功能，请按以下步骤操作：

1. 在 `TestRunner` 类中添加新的测试方法：

```python
def test_new_feature(self):
    """测试新功能"""
    print("=== 新功能测试 ===")
    try:
        # 测试逻辑
        print("✅ 新功能测试通过")
        return True
    except Exception as e:
        print(f"❌ 新功能测试失败: {e}")
        return False
```

2. 在 `__init__` 方法中注册新测试：

```python
self.tests: Dict[str, Callable] = {
    # ... 现有测试
    'new_feature': self.test_new_feature,
    # ...
}
```

3. 更新 `run_all_tests` 方法中的测试列表：

```python
test_methods = [
    # ... 现有测试
    ('新功能', self.test_new_feature),
    # ...
]
```

## 测试输出格式

测试脚本使用统一的输出格式：

- ✅ 表示成功/通过
- ❌ 表示失败/错误
- ⚠️ 表示警告/部分成功
- 📥 表示加载/初始化
- 📁 表示文件/目录操作
- 🤖 表示模型相关操作
- 🔧 表示配置/设置操作
- 🎉 表示完成/成功

## 注意事项

1. **环境要求**：确保在项目根目录下运行测试
2. **依赖检查**：某些测试需要特定的依赖包
3. **资源限制**：模型推理测试可能需要较多内存
4. **网络连接**：下载相关测试需要网络连接
5. **权限要求**：某些测试可能需要文件读写权限

## 故障排除

### 常见问题

1. **模块导入失败**
   - 检查是否在项目根目录下运行
   - 确认所需依赖包已安装

2. **模型文件不存在**
   - 运行模型下载测试检查文件状态
   - 确认模型路径配置正确

3. **权限错误**
   - 检查文件和目录的读写权限
   - 确认有足够的磁盘空间

### 调试模式

如果测试失败，可以查看详细的错误信息和堆栈跟踪。测试脚本会自动显示异常详情。

## 维护指南

1. **定期运行**：建议在每次代码更改后运行相关测试
2. **更新测试**：当添加新功能时，及时添加对应测试
3. **清理输出**：保持测试输出简洁明了
4. **文档同步**：更新测试时同步更新本文档

## 版本历史

- v1.0.0 - 初始版本，整合所有现有测试功能