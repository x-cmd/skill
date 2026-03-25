---
name: mistral
description: >
  集成 Mistral AI，用于高效的语言建模和文本生成任务。
  核心场景：当用户想要通过 CLI 使用 Mistral 模型进行聊天或翻译时。
license: MIT
---

# mistral - Mistral AI 集成

`mistral` 模块为 Mistral AI 的服务提供命令行界面，支持高效的文本生成和多语言翻译。

## 激活时机
- 当用户想要与 Mistral 模型聊天时。
- 当用户需要使用 Mistral API 翻译文件内容时。
- 当检查或管理 Mistral API 密钥时。

## 核心原则与规范
- **API 密钥管理**: 使用 `init` 或 `--cfg apikey=<key>` 进行设置。
- **别名访问**: 使用 `@mistral` 别名进行快速交互。
- **探索模型**: 使用 `model ls` 查看所有可用的 Mistral 模型。

## 实战示例

### 翻译文件
```bash
# 使用 Mistral 将文件翻译为中文
@mistral --file ./document.en.md "翻译为中文"
```

### 列出模型
```bash
# 查看所有支持的 Mistral 模型
x mistral model ls
```

## 交付验证清单
- [ ] 确保已配置 Mistral API 密钥。
- [ ] 使用 `--file` 标志时验证文件路径。
