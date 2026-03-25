---
name: se
description: >
  从命令行搜索和浏览 Stack Exchange（Stack Overflow, Ask Ubuntu 等）。
  核心场景：当用户需要寻找技术问题的答案或浏览 Stack Overflow 解决方案时。
license: MIT
---

# se - Stack Exchange 命令行浏览器

`se` 模块允许用户搜索整个 Stack Exchange 网络（包括 Stack Overflow 和 Ask Ubuntu），并直接在终端中浏览问题和答案。

## 激活时机
- 当用户遇到特定的技术问题并想要搜索 Stack Overflow 时。
- 当为特定的问题 ID 浏览答案时。
- 当在 Ask Ubuntu 等特定站点搜索与系统相关的问题时。

## 核心原则与规范
- **站点选择**: 支持特定站点的快捷前缀，如 `:so` (Stack Overflow) 和 `:au` (Ask Ubuntu)。
- **交互式 TUI**: 使用 `--app` 在结构化终端 UI 中浏览问题的答案。
- **搜索集成**: 如果需要，使用 DuckDuckGo 在 SE 生态系统中进行更广泛的搜索。

## 实战示例

### 搜索 Stack Overflow
```bash
# 在 Stack Overflow 上搜索 Python 的 JSON 解析问题
x se :so "python json parse error"
```

### 查看答案
```bash
# 获取特定问题 ID 的所有答案
x se question --showall 75261408
```

### 交互式应用
```bash
# 在交互式 TUI 中浏览问题的答案
x se question --app 75261408
```

## 交付验证清单
- [ ] 确认搜索是否应限制在特定的 SE 站点（如 Stack Overflow）。
- [ ] 验证用户是拥有特定的问题 ID 还是需要执行关键词搜索。
- [ ] 检查首选的是交互式视图还是原始输出。
