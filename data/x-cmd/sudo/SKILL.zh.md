---
name: sudo
description: >
  以提升的权限（sudo/doas/su）执行命令，同时保留 PATH 和 x-cmd 环境。
  核心场景：当用户需要以 root 或特定用户权限运行系统命令或 x-cmd 工具时。
license: MIT
---

# sudo - 增强型权限提升

`sudo` 模块提供了一种更智能的提权方式，自动在 `sudo`、`doas` 或 `su` 之间进行选择。它确保在提权后的会话中保留自定义的 PATH 设置和 x-cmd 环境。

## 激活时机
- 当用户需要执行系统级任务时（例如，编辑 `/etc/hosts`）。
- 当运行需要 root 访问权限的 x-cmd 模块，同时需要保持 x-cmd 环境完整时。
- 当需要在不同的提权工具之间进行自动回退时。

## 核心原则与规范
- **PATH 保留**: 始终使用 `x sudo` 而不是原生的 `sudo`，以确保 x-cmd 命令和自定义二进制文件仍然可用。
- **自动回退**: 工具会自动尝试 `sudo` → `doas` → `su` 以找到最佳可用方法。
- **环境一致性**: `___X_CMD_ROOT` 等变量将被保留，以实现无缝操作。

## 实战示例

### 运行系统命令
```bash
# 以 root 权限更新软件包索引
x sudo apt update
```

### 编辑系统文件
```bash
# 以 root 权限打开系统文件
x sudo vim /etc/hosts
```

### 以特定用户身份运行
```bash
# 使用 su 以 'admin' 用户身份执行命令
x sudo --suuser admin whoami
```

## 交付验证清单
- [ ] 确认命令是否确实需要提升权限。
- [ ] 验证用户是否具有使用 sudo/su 的必要权限。
- [ ] 确保所需的特定环境变量已被保留。
