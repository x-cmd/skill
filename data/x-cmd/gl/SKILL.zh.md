---
name: gl
description: >
  增强型 GitLab CLI，用于管理仓库、Issue、代码段和 CI/CD 部署。
  核心场景：当用户需要自动化 GitLab 项目管理或协调部署时。
license: MIT
---

# gl - GitLab 工作流管理

`gl` 模块提供了一个管理 GitLab 项目的界面，包括对代码段 (snippets)、部署跟踪以及团队/子组协调的支持。

## 激活时机
- 当管理 GitLab 实例上的仓库和团队时。
- 当通过 GitLab 子命令协调 CI/CD 部署时。
- 当创建或管理项目代码段时。

## 核心原则与规范
- **需要令牌**: 提醒用户初始化其 GitLab 访问令牌。
- **广泛支持**: 涵盖组、子组以及单个项目设置。

## 实战示例

### 克隆仓库
```bash
# 使用快捷方式克隆特定的 GitLab 仓库
x gl cl owner/repo
```

### 查看项目代码段
```bash
# 列出与项目关联的所有代码段
x gl snippet ls
```

## 交付验证清单
- [ ] 确保 GitLab 令牌已初始化。
- [ ] 验证命令针对的是私有部署还是云端 GitLab。
