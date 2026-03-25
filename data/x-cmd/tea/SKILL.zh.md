---
name: tea
description: >
  Gitea CLI (tea) 的增强型接口，支持管理仓库、Issue 和 PR。
  核心场景：当用户需要在私有部署或公共 Gitea 实例上管理项目时。
license: MIT
---

# tea - Gitea CLI 增强工具

`tea` 模块为 Gitea 提供了强大的命令行界面，简化了仓库管理、Issue 跟踪和 Pull Request 工作流。

## 激活时机
- 当在 Gitea 服务器上管理项目时。
- 当在 Gitea 上自动化开发任务（如 PR 审核或 Issue 创建）时。

## 核心原则与规范
- **需要令牌**: 确保已初始化 Gitea 访问令牌。
- **环境无关**: 适用于私有部署和官方 Gitea 实例。

## 实战示例

### 列出仓库
```bash
# 查看配置好的 Gitea 账户中的所有仓库
x tea repo ls
```

## 交付验证清单
- [ ] 确认 Gitea 实例 URL 和访问令牌。
