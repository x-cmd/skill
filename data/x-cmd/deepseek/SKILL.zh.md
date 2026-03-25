---
name: deepseek
description: >
  集成 DeepSeek AI，用于高性能文本生成和专业推理任务。
  核心场景：当用户想要使用 DeepSeek 的 V3 或推理模型进行编码、翻译或复杂逻辑处理时。
license: MIT
---

# deepseek - DeepSeek AI 集成

`deepseek` 模块提供 DeepSeek AI 模型的命令行界面，包括强大的 V3 模型和专注于推理的模型。

## 激活时机
- 当用户想要使用 DeepSeek 进行聊天、文本生成或编码任务时。
- 当用户需要使用 `deepseek-reasoner` 模型进行复杂推理或数学解题时。
- 当检查 DeepSeek 账户余额或管理 API 密钥时。

## 核心原则与规范
- **API 密钥管理**: 使用 `init` 或 `--cfg apikey=<key>` 进行设置。
- **模型选择**: 默认通常为 V3 (`@ds`)。对于需要深度思考的任务，请使用 `--model deepseek-reasoner`。
- **余额监控**: 使用 `balance` 子命令跟踪信用额度。

## 补充场景
- **快捷别名**: 使用 `@ds` 进行快速交互。
- **项目配置**: 使用 `--cfg` 选项按项目设置默认模型。

## 实战示例

### 与 DeepSeek V3 聊天
```bash
# 提出一般性问题
@ds "解释量子纠缠的概念"
```

### 逻辑推理
```bash
# 使用推理模型处理复杂逻辑
@ds --model deepseek-reasoner "解决这个复杂的概率问题：..."
```

### 检查账户余额
```bash
# 查看当前余额
x deepseek balance
```

## 交付验证清单
- [ ] 确保已初始化 DeepSeek API 密钥。
- [ ] 确认推理模型是否更适合当前任务。
