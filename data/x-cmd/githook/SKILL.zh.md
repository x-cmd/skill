---
name: githook
description: >
  高效地管理 Git 钩子，支持初始化、罗列和移除。
  核心场景：当用户需要为自动化和策略执行设置或审计 Git 钩子时。
license: MIT
---

# githook - Git 钩子管理工具

`githook` 模块提供了一个简单的界面来管理仓库内的 Git 钩子，使用户能够轻松地安装、查看或移除用于 Lint 或自动化测试等任务的钩子。

## 激活时机
- 当在仓库中设置 pre-commit 或 post-merge 钩子时。
- 当审计现有钩子以了解自动化的仓库行为时。

## 实战示例

### 列出钩子
```bash
# 查看 Git 仓库中当前激活的所有钩子
x githook ls
```

## 交付验证清单
- [ ] 确认仓库路径。
- [ ] 验证具体的钩子类型（pre-commit 等）。
