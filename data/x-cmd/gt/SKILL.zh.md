---
name: gt
description: >
  增强型 Gitee CLI，用于在 Gitee 平台上管理仓库、Issue 和 PR。
  核心场景：当用户需要自动化开发工作流或管理 Gitee 上的项目时。
license: MIT
---

# gt - Gitee 工作流管理

`gt` 模块提供了管理 Gitee 活动的界面，支持仓库生命周期管理、Issue 跟踪以及 Pull Request 协调。

## 激活时机
- 当管理托管在 Gitee 平台上的项目时。
- 当在基于 Gitee 的团队中自动化 PR 审核或 Issue 跟踪时。
- 当管理 Gitee 组织或企业级设置时。

## 核心原则与规范
- **需要令牌**: 使用 `init` 或 `--cfg` 设置 Gitee 个人访问令牌。
- **快捷方式支持**: 使用 `cl` 作为 `repo clone` 的快捷方式。

## 实战示例

### 查看用户信息
```bash
# 从 Gitee 检索当前用户信息
x gt user info
```

### 列出仓库 (交互式)
```bash
# 在 TUI 表格中查看并浏览你的 Gitee 仓库
x gt repo ls
```

## 交付验证清单
- [ ] 确认 Gitee 令牌已配置。
- [ ] 验证项目是个人账户还是企业账户的一部分。
