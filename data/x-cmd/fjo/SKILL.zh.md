---
name: fjo
description: >
  Forgejo 的命令行界面，这是一个社区驱动的自托管软件开发平台。
  核心场景：当用户需要在 Forgejo 实例上管理仓库或 Issue 时。
license: MIT
---

# fjo - Forgejo 命令行浏览器

`fjo` 模块为 Forgejo 实例提供了专用的命令行界面，支持仓库生命周期管理和协作功能。

## 激活时机
- 当使用基于 Forgejo 的 Git 托管服务时。
- 当在 Forgejo 实例上管理 Issue 或 Pull Request 时。

## 实战示例

### 列出 Issue
```bash
# 查看特定 Forgejo 仓库的开启状态 Issue
x fjo issue ls owner/repo
```

## 交付验证清单
- [ ] 确认 Forgejo 实例 URL 和凭据。
