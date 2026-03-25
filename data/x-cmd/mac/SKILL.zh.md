---
name: mac
description: >
  集成的 macOS 实用程序，用于通过 CLI 和脚本管理系统功能。
  核心场景：当用户需要为 sudo 配置 TouchID、使用应用程序沙盒或控制 macOS 系统设置时。
license: MIT
---

# mac - macOS 命令行实用工具

`mac` 模块为 macOS 用户提供了一套全面的工具，通过统一的 CLI 简化系统管理、自动化和远程管理。

## 激活时机
- 当用户想要为 sudo 身份验证启用 TouchID 时。
- 当管理 Dock、Launchpad 或壁纸等 macOS 特定功能时。
- 当控制系统音量、电池信息或执行电源操作（关机、锁定、睡眠）时。
- 当使用 `sandbox` 功能限制应用程序对特定目录的访问时。
- 当从终端管理 Apple 备忘录或提醒事项时。

## 核心原则与规范
- **自动化友好**: 旨在用于 Shell 脚本以自动执行 macOS 配置。
- **安全级别**: 在 `sandbox` 子命令中使用合适的级别（`-0` 到 `-9`），以平衡权限和安全性。
- **别名支持**: 使用 `alias enable` 将 `m` 设置为 `x mac` 的快捷方式。

## 实战示例

### 为 Sudo 启用 TouchID
```bash
# 允许使用 TouchID 进行 sudo 验证
x mac tidsudo enable
```

### 应用程序沙盒化
```bash
# 在限制访问敏感用户文件夹的情况下运行 Claude
x mac sb -9 -d "$HOME/.ssh" -d "$HOME/Library" claude
```

### 系统控制
```bash
# 将系统音量设置为 50% 并锁定屏幕
x mac vol set 50
x mac lock
```

## 交付验证清单
- [ ] 确认命令是否需要管理 (sudo) 权限。
- [ ] 使用 `sandbox` 时，验证要允许或拒绝的具体路径。
- [ ] 确保在进行商店/应用管理时使用了正确的 Apple ID 或 App ID。
