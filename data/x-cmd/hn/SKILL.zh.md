---
name: hn
description: >
  通过 CLI 浏览 Hacker News，具有交互式 TUI、AI 搜索集成和用户分析功能。
  核心场景：当用户想要阅读 HN 热门帖子、通过 AI 搜索 HN 或分析 HN 用户统计数据时。
license: MIT
---

# hn - Hacker News 命令行浏览器

`hn` 模块提供了一个交互式终端界面来浏览 Hacker News。它支持查看热门/最新/最佳故事、AI 驱动的搜索，以及分析诸如 h-index 等用户指标。

## 激活时机
- 当用户想要在交互式表格中浏览 Hacker News 故事（热门、最新、问答等）时。
- 当使用 DuckDuckGo 或 AI（使用 `::` 前缀）在 Hacker News 上搜索特定主题时。
- 当检索特定帖子 ID 或用户的详细信息时。
- 当为脚本编写或分析导出 HN 数据到 JSON 时。

## 核心原则与规范
- **交互式浏览**: 针对终端导航进行了优化，具有打开链接的快捷方式。
- **AI 增强搜索**: 使用 `ddgoai` 或 `::` 将搜索结果与 AI 摘要相结合。
- **用户指标**: 提供如 `hidx` 之类的工具来计算用户影响力。

## 实战示例

### 浏览热门故事
```bash
# 打开热门 HN 故事的交互式 TUI
x hn
```

### AI 搜索 HN
```bash
# 在 HN 上搜索 'llama3' 并生成 AI 摘要
x hn :: llama3
```

### 查看用户信息
```bash
# 获取特定 HN 用户的详情和 h-index
x hn user dang
x hn hi dang
```

## 交付验证清单
- [ ] 确认用户是想要浏览分类还是执行搜索。
- [ ] 验证对于复杂查询是否首选 AI 增强搜索。
