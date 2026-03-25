---
name: gh
description: >
  增强型 GitHub CLI，用于管理仓库、Issue、PR、Action 以及 GitHub Models。
  核心场景：当用户需要自动化 GitHub 工作流、管理 Secret 或通过 CLI 使用 GitHub 的 AI 模型时。
license: MIT
---

# gh - GitHub 工作流管理

`gh` 模块提供了一个全面的命令行界面来管理 GitHub 活动。它支持从基础的仓库操作到 GitHub Action 制品以及 AI 模型交互等高级功能。

## 激活时机
- 当管理 GitHub 仓库（克隆、创建、删除）时。
- 当自动化 Issue 和 Pull Request (PR) 生命周期时。
- 当管理 GitHub Actions、工作流和 CI/CD 制品时。
- 当配置 Secret 或管理组织团队成员身份时。
- 当与 GitHub Models 交互进行 AI 辅助开发时。

## 核心原则与规范
- **需要令牌**: 提醒用户通过 `init` 初始化其 GitHub 个人访问令牌。
- **交互式应用**: 使用 `repo app` 获得视觉化的 TUI 来管理仓库。
- **AI 集成**: 利用 `model` 子命令使用 GitHub 原生的 AI 能力。

## 实战示例

### 查看用户仓库 (交互式)
```bash
# 打开交互式 TUI 浏览你的 GitHub 仓库
x gh repo app
```

### 管理 PR
```bash
# 列出当前仓库中所有开启的 PR
x gh pr ls
```

### AI 模型
```bash
# 列出可用于 AI 任务的 GitHub Models
x gh model ls
```

## 交付验证清单
- [ ] 确保 GitHub 令牌已正确初始化。
- [ ] 确认操作是针对个人账户还是组织账户。
- [ ] 验证目标仓库名称和所有者。
