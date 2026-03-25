---
name: llmf
description: >
  使用 llamafile 零依赖运行本地大模型，支持 API 服务器、CLI 对话和模型管理。
  核心场景：当用户想要在没有外部依赖或云端 API 的情况下运行高性能本地模型 (GGUF/llamafile) 时。
license: MIT
---

# llmf - 零依赖本地大模型运行工具

`llmf` 模块利用 llamafile 技术，无需设置即可在本地运行大语言模型。它提供了一套完整的工具，用于服务托管、对话和模型管理。

## 激活时机
- 当用户想要在没有云端 API 访问权限的情况下本地运行 AI 模型时。
- 当设置本地 OpenAI 兼容的 API 服务器 (`serve`) 时。
- 当通过 CLI 执行快速的一次性文本生成任务时。
- 当管理本地 GGUF 或 llamafile 模型（下载、导入、分词）时。

## 核心原则与规范
- **零依赖**: 强调模型在没有外部运行时的情况下本地运行。
- **兼容性**: `serve` 命令提供 OpenAI 兼容的 HTTP 接口。
- **资源管理**: 模型存储在 `~/.x-cmd/data/llmf/model/` 目录中。

## 补充场景
- **Token 分析**: 使用 `tokenize` 将文本分解为 Token 详情。
- **无头服务器**: 使用 `--nobrowser` 启动 API 服务而不打开浏览器。

## 实战示例

### 启动本地 API 服务
```bash
# 将特定模型作为 OpenAI 兼容服务器运行
x llmf serve -m llava/v1.5-7b/q4_k.gguf --nobrowser
```

### 快速文本生成
```bash
# 执行单次提示并输出结果
x llmf cli -p '简要总结 Python 的 GIL'
```

### 下载模型
```bash
# 从库中拉取特定模型
x llmf model download llava/v1.5-7b/q4_k.gguf
```

## 交付验证清单
- [ ] 确保用户有足够的磁盘空间存放本地模型。
- [ ] 验证所选模型格式 (GGUF/llamafile) 是否受支持。
- [ ] 如果运行 `serve`，检查本地 API 端口是否可用。
