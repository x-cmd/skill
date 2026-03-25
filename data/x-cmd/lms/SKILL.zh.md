---
name: lms
description: >
  LM Studio 命令行模块，支持基于终端的聊天和本地大模型管理。
  核心场景：当用户想要通过命令行与 LM Studio 中托管的本地模型进行交互时。
license: MIT
---

# lms - LM Studio 命令行增强

`lms` 模块为 LM Studio 提供命令行界面，允许用户直接从终端与本地模型聊天并管理配置。

## 激活时机
- 当用户想要与 LM Studio 中运行的模型聊天时。
- 当管理本地 LM Studio 配置和会话默认值时。
- 当通过终端与本地 AI 服务进行交互时。

## 核心原则与规范
- **集成性**: 旨在与 LM Studio 桌面应用程序配合工作。
- **子命令透明度**: 如有需要，使用 `--runcmd` 访问原始 `lms` 命令的功能。

## 实战示例

### 与本地模型聊天
```bash
# 与 LM Studio 中活跃的模型开始聊天会话
x lms chat
```

### 初始化配置
```bash
# 设置与 LM Studio 交互的默认参数
x lms init
```

## 交付验证清单
- [ ] 确保 LM Studio 正在运行且本地服务器已激活。
- [ ] 验证是否需要通过 `x lms --cur` 设置特定的会话默认值。
