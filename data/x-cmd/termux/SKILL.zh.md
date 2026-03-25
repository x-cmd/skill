---
name: termux
description: >
  全面的 Termux Android 实用工具，提供包管理、API 集成和 PRoot 发行版。
  核心场景：当用户需要与 Android 硬件交互（短信、震动、朗读）或通过 PRoot 管理 Linux 发行版时。
license: MIT
---

# termux - Termux Android 增强工具

`termux` 模块将各种 Android 特定的功能集成到 Termux 终端中，允许进行硬件交互并轻松管理隔离的 Linux 环境。

## 激活时机
- 当用户想要与 Android 传感器或硬件（电池、震动、音量）交互时。
- 当通过 CLI 进行短信管理（读取或发送消息）时。
- 当在 Android 设备上使用文本转语音 (`say`) 功能时。
- 当在没有 root 权限的情况下，使用 PRoot 安装或管理 Linux 发行版（Ubuntu, Alpine, Debian）时。

## 核心原则与规范
- **API 依赖**: 提醒用户必须安装 `termux-api` 才能使用硬件相关的子命令。
- **免 Root 环境**: 强调 `proot-distro` 允许在不 root 设备的情况下运行完整的 Linux 环境。
- **快捷方式**: 支持 `m` 别名（`x termux` 的默认快捷方式）。

## 实战示例

### 硬件交互
```bash
# 让手机震动 500 毫秒
x termux vibrate 500
# 使用 Android 语音引擎朗读文本
x termux say "系统更新完成"
```

### PRoot 管理
```bash
# 安装并运行 Ubuntu 环境
x termux pd install ubuntu
x termux ubu bash
```

### 短信管理
```bash
# 向特定号码发送短信
x termux sms send -n 123456789 "来自 x-cmd 的问候"
```

## 交付验证清单
- [ ] 验证执行硬件任务时是否已安装 `termux-api` 包。
- [ ] 确认 PRoot 命令的目标 Linux 发行版名称。
- [ ] 确保 Android 设备已向 Termux 授予必要的权限。
