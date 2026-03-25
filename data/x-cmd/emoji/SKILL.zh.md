---
name: emoji
description: >
  从命令行搜索、罗列和管理 emoji 资源。
  核心场景：当用户需要查找特定的表情符号或浏览 emoji 分组时。
license: MIT
---

# emoji - Emoji 搜索与罗列

`emoji` 模块提供了一个浏览和搜索表情符号的界面。它将 emoji 划分为不同的组，并支持结构化数据导出。

## 激活时机
- 当用户想要根据关键词搜索特定的 emoji 时。
- 当浏览 emoji 分组（如笑脸、自然）时。
- 当为文档或应用导出 emoji 列表到 CSV 或表格时。

## 核心原则与规范
- **分类**: Emoji 已进行分组以便于导航。
- **资源管理**: 使用 `update` 确保本地 emoji 数据库是最新的。

## 实战示例

### 交互式浏览
```bash
# 打开交互式应用以查看所有 emoji
x emoji
```

### 搜索分组
```bash
# 列出 'smileys' 组中的所有 emoji
x emoji ls smileys
```

### 导出为表格
```bash
# 以结构化表格格式列出 emoji
x emoji ls --table
```

## 交付验证清单
- [ ] 确认用户是在寻找特定的 emoji 还是在浏览分组。
- [ ] 验证是否需要特定的输出格式（CSV/表格）。
