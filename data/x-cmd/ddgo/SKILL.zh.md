---
name: ddgo
description: >
  DuckDuckGo 搜索的命令行工具，支持结果总结和 AI 驱动的答案提取。
  核心场景：当用户需要执行网页搜索或从 DuckDuckGo 结果中获取 AI 总结的答案时。
license: MIT
---

# ddgo - DuckDuckGo 命令行搜索

`ddgo` 模块为 DuckDuckGo 提供了一个强大的基于终端的界面，使用户能够执行网页搜索、以结构化格式提取结果，并利用 AI 总结答案。

## 激活时机
- 当用户想要通过 CLI 执行注重隐私的网页搜索时。
- 当针对特定查询需要 AI 生成的搜索结果总结时。
- 当为进一步处理导出搜索结果到 JSON 时。
- 当执行特定站点的搜索时（例如 `site:x-cmd.com`）。

## 核心原则与规范
- **AI 集成**: 使用 `--ai` 标志自动选择并总结最相关的搜索结果。
- **数据友好**: 支持结构化 JSON 输出以便脚本集成。
- **结构化查看**: 使用 `dump --app` 以交互式表格视图查看搜索结果。

## 实战示例

### AI 总结搜索
```bash
# 获取关于 'bash tips' 搜索结果的 AI 总结
x ddgo --ai "bash best practices"
```

### 特定站点搜索
```bash
# 在 x-cmd.com 网站上搜索 'jq' 内容
x ddgo dump --json "site:x-cmd.com jq"
```

### 前几条结果
```bash
# 检索查询的前 10 条搜索结果
x ddgo --top 10 "python threading tutorial"
```

## 交付验证清单
- [ ] 确认用户需要的是一般搜索还是 AI 总结的回答。
- [ ] 验证结果是否应限制在特定的站点或数量内。
- [ ] 检查后续任务是否需要 JSON 输出。
