---
name: gemini
description: >
  集成 Google Gemini AI 进行聊天、文本分析及图片理解等多模态任务。
  核心场景：当用户需要使用 Gemini 模型进行高级聊天、文件上下文翻译或通过 CLI 分析图片时。
license: MIT
---

# gemini - Google Gemini AI 集成

`gemini` 模块提供 Google Gemini AI 模型的强大命令行界面，支持文本生成、基于文件的上下文及图片等多模态输入。

## 激活时机
- 当用户想要使用 Gemini 进行聊天或复杂的文本生成时。
- 当用户需要使用 Gemini 的大上下文窗口分析或翻译文件时。
- 当用户想要通过终端提供图片供 AI 分析时。
- 当管理 Gemini API 密钥或探索可用模型时。
- 当使用 Gemini 集成的 Google Search 工具 (`gg`) 时。

## 核心原则与规范
- **API 密钥管理**: 使用 `init` 或 `--cfg apikey=<key>` 进行设置。
- **多模态支持**: 使用 `--file` 标志附加图片或文档供 AI 处理。
- **Token 计数**: 使用 `--count` 标志在请求前或请求期间估算 Token 使用情况。
- **Google 搜索集成**: 利用 `gg` 子命令在 AI 响应中利用来自 Google Search 的实时信息。

## 补充场景
- **Git 提交信息**: 将 `git diff` 传送到 `@gemini` 以获得高质量、标准化的提交信息。
- **组合工具**: 将来自其他工具（例如，Wikipedia 的 `x wkp`）的输出传送到 `@gemini` 进行专业分析。
- **官方 CLI**: 通过 `x gemini cli` 访问 Google 官方 Gemini CLI。

## 实战示例

### 带有图片和文本的聊天
```bash
# 请求 Gemini 描述图片文件
@gemini --file ./pic.jpg "这张图片描述了什么？"
```

### 翻译文件
```bash
# 使用 Gemini 的上下文翻译多个本地文件
x gemini chat request --file ./abstract.en.md --file ./content.en.md "将这些翻译为中文"
```

### Google 搜索集成
```bash
# 提出问题并使用 Google 搜索获取答案
x gemini gg "x-cmd 的最新功能是什么？"
```

## 交付验证清单
- [ ] 确保已使用 `x gemini init` 配置了 Gemini API 密钥。
- [ ] 验证附加的文件（图片/文档）是否存在且为支持的格式。
- [ ] 如果需要，确认具体的模型版本。
