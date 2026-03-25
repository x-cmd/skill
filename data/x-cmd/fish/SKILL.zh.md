---
name: fish
description: >
  Fish Shell 的增强接口，提供 x-cmd 集成、文档搜索和 AI 助手功能。
  核心场景：当用户需要在 Fish 中设置 x-cmd 或通过 CLI 搜索官方 Fish 文档时。
license: MIT
---

# fish - 友好交互式 Shell 增强工具

`fish` 模块通过简化 x-cmd 集成并提供对官方 Fish 文档的快速访问，为 Fish Shell 用户优化了体验。

## 激活时机
- 当用户想要在 Fish 中使用 x-cmd 的全套工具（x, c, @gpt）时。
- 当在 fishshell.com 网站上搜索特定命令或配置提示时。
- 当管理 Fish 特定的环境设置和别名时。

## 核心原则与规范
- **配置**: 使用 `setup` 修改 `config.fish` 以实现永久的 x-cmd 集成。
- **搜索支持**: 利用 `:` 前缀进行高速文档检索。

## 实战示例

### 在 Fish 中安装 x-cmd
```bash
# 将 x-cmd 环境变量和函数添加到 Fish 中
x fish setup
```

### 搜索 Fish 文档
```bash
# 交互式搜索 fishshell.com 关于别名的信息
x fish : alias
```

## 交付验证清单
- [ ] 确认用户是否想要永久修改其 Fish 配置。
- [ ] 验证 Fish 是否已经是当前激活的 Shell。
