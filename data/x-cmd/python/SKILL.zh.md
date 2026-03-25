---
name: python
description: >
  增强型 Python 环境管理器，提供版本管理、pip 集成和自动设置。
  核心场景：当用户需要安装特定的 Python 版本、管理包或运行脚本时。
license: MIT
---

# python - Python 环境管理

`python` 模块提供了一个统一的命令行界面来管理 Python 版本和环境，并与 x-cmd 的包系统集成，以便在不同的操作系统平台上实现无缝设置。

## 激活时机
- 当安装或在不同的 Python 版本之间切换时。
- 当使用集成的 `pip` 功能管理 Python 包时。
- 当运行 Python 脚本或进入交互式 REPL 时。

## 核心原则与规范
- **集成性**: 与 asdf 及其他版本管理器配合工作，以确保可用性。
- **子模块支持**: 使用 `pip` 管理当前 Python 环境中的依赖项。

## 实战示例

### 安装包
```bash
# 在受管 Python 环境中全局安装一个包
x python pip install requests
```

### 运行脚本
```bash
# 使用受管运行时执行 Python 脚本
x python ./script.py
```

## 交付验证清单
- [ ] 如果有多个可用版本，确认目标 Python 版本。
- [ ] 验证用于 pip 安装的包名。
