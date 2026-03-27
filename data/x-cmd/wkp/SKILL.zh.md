---
name: wkp
description: >
  维基百科搜索与摘要提取工具。
  核心场景：当 AI 需要在终端中快速获取某项技术的定义或某个概念的官方摘要时。
license: MIT
---

# x wkp - 维基百科助手 (AI 优化版)

`x wkp` 提供了极简的接口，通过命令行获取维基百科的文章列表、建议及详细摘要。

## 激活时机
- 当需要快速查询某个术语、历史事件或技术概念的定义时。
- 当需要获取特定 Wikipedia 条目的纯文本摘要时。
- 当需要通过关键词获取搜索建议或相关词条列表时。

## 核心原则与规范
- **非交互优先**: 避免使用 `--app` 交互界面，直接使用 `extract` 或 `hop` 子命令获取纯文本。
- **结构化获取**: 优先使用 `extract` 获取详细正文，或者 `hop` 获取首位匹配项的精简摘要。

## 实战示例

### 获取条目的详细摘要
```bash
# 获取 "OpenAI" 的详细中文摘要
x wkp extract OpenAI
```

### 搜索相关条目
```bash
# 搜索关键词 "Large Language Model"
x wkp search "Large Language Model"
```

### 直接获取首个结果的摘要
```bash
# 一次性搜索并输出第一个匹配结果的摘要 (最高效)
x wkp hop "Rust Programming"
```

### 获取搜索建议
```bash
# 当不确定准确拼写时获取相关词条建议
x wkp suggest "Quantom Computing"
```

## 交付验证清单
- [ ] 确认查询词是否需要引号包裹。
- [ ] 根据需要选择 `search` (列表) 或 `extract` (内容)。
- [ ] 默认使用英语和中文查询以获取最全面的信息。
