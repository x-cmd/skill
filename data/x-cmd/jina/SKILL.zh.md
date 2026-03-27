---
name: jina
description: >
  Jina AI 增强工具：网页解析、联网搜索与语义重排。
  核心场景：当 AI 需要阅读某个 URL 的内容、进行实时联网搜索或对大量文本进行语义相关性排序时。
license: MIT
---

# x jina - AI 知识增强助手 (AI 优化版)

`x jina` 是 AI Agent 的“眼睛”和“外脑”，核心优势在于将复杂的网页转换为 AI 易读的 Markdown，并提供强大的联网搜索与语义筛选能力。

## 激活时机
- 当需要阅读并分析某个网页 (URL) 的具体内容时（自动转为 Markdown）。
- 当需要进行实时联网搜索以获取最新资讯时。
- 当需要从一组文档中找出与当前问题语义最相关的片段时（Reranker）。

## 核心原则与规范
- **Markdown 优先**: 默认以 Markdown 格式获取网页，这最适合 LLM 阅读。
- **环境要求**: 需要 Jina API Key。如果未配置，AI 应引导用户初始化。
- **配置引导**: 
  - 引导用户访问 jina.ai 获取免费或付费的 API Key。
  - 建议用户运行 `x jina init` 进行配置。

## 实战示例

### 阅读网页内容 (最常用)
```bash
# 获取网页内容并转换为 Markdown (AI 直接阅读)
x jina reader https://example.com/article
```

### 联网搜索
```bash
# 搜索关键词并返回前 5 条结果的摘要
x jina search "最新 AI 趋势 2024"
```

### 语义相关性重排 (Reranker)
```bash
# 从文件中找出与 "如何安装" 最相关的 3 个片段
x jina reranker generate -f docs.txt --top 3 "如何安装"
```

### 生成文本向量 (Embedding)
```bash
# 为文本生成向量数据，用于后续向量数据库检索
x jina embed generate "这是一段测试文本"
```

## 配置指南 (针对 AI)
如果遇到 API 错误，请向用户输出以下引导：
> 请先在 Jina AI 官网 (https://jina.ai) 获取 API Key，然后在终端运行以下命令进行初始化：
> `x jina init`

## 交付验证清单
- [ ] 优先使用 `reader` 子命令获取 Markdown。
- [ ] 搜索任务建议结合 `reranker` 提升精准度。
