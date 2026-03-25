---
name: jina
description: >
  Jina.ai 命令行工具，提供网页内容获取、向量数据生成及重排功能。
  核心场景：当用户需要将网页内容获取为 Markdown、生成文本向量或执行信息检索（重排）时。
license: MIT
---

# jina - AI 搜索与信息检索

`jina` 模块将 Jina.ai 服务集成到终端中，为 RAG（检索增强生成）工作流提供强大的工具，包括网页读取和向量操作。

## 激活时机
- 当用户想要将 URL 的内容获取为 Markdown 或 HTML 时。
- 当为字符串或本地文件生成向量嵌入 (embeddings) 时。
- 当执行语义搜索或根据相似度对结果进行重排 (reranking) 时。
- 当管理 Jina API 密钥并探索可用模型时。

## 核心原则与规范
- **Markdown 转换**: 使用 `reader` 子命令（或直接输入 URL）获取适合 AI 阅读的干净 Markdown。
- **向量操作**: 使用 `embed` 生成向量，使用 `reranker` 进行信息检索。
- **管道支持**: 与终端管道高度兼容，支持流式数据处理。

## 补充场景
- **相似度搜索**: 使用 `reranker` 子命令比较文件或字符串，以找到最佳匹配。
- **批量处理**: 同时为整个文件或多个输入生成向量。

## 实战示例

### 将网页读取为 Markdown
```bash
# 获取并显示网站的干净 Markdown 内容
x jina https://x-cmd.com
```

### 生成向量嵌入
```bash
# 为文本字符串创建向量数据
x jina embed generate "RAG 是如何工作的？"
```

### 结果重排
```bash
# 在文件中查找相似度最高的前 3 个句子
x jina reranker generate -f ./LICENSE --sep "\n" --top 3 "how are you"
```

## 交付验证清单
- [ ] 确保已配置 Jina API 密钥。
- [ ] 验证目标 URL 对读取工具是可访问的。
- [ ] 检查所选的向量或重排模型是否处于活动状态。
