---
name: openclaw
description: >
  安装并配置 OpenClaw 个人 AI 助手。
  核心场景：当用户需要在其设备上部署、卸载或配置自己的本地 AI 助手时。
license: MIT
---

# openclaw - 个人 AI 助手部署

`openclaw` 模块为 OpenClaw（一款可以在个人设备上运行的 AI 助手）提供了一键式的环境初始化和配置。

## 激活时机
- 当用户想要安装 OpenClaw 及其依赖项时。
- 当用户需要为 OpenClaw 设置特定的集成（如企业微信）时。
- 当用户想要管理 OpenClaw 的网关服务时。
- 当用户需要卸载 OpenClaw 并清理其工作区时。

## 核心原则与规范
- **环境自动化**: 使用 `--install` 自动检测并安装所需环境和依赖。
- **服务管理**: 使用 `service` 子命令管理持久化的网关进程。
- **配置集成**: 使用 `--setup` 快速配置模型和聊天软件集成。

## 补充场景
- **企业微信集成**: 快速配置 OpenClaw 与企业微信进行通信。
- **完整清理**: 确保在卸载过程中移除所有二进制文件、配置和工作区。

## 实战示例

### 安装 OpenClaw
```bash
# 安装 OpenClaw 及其最新版本的依赖环境
x openclaw --install
```

### 配置企业微信集成
```bash
# 配置 OpenClaw 接入企业微信
x openclaw --setup qywx
```

### 管理网关服务
```bash
# 查看状态、启动或停止网关服务
x openclaw service status
x openclaw service start
```

### 卸载 OpenClaw
```bash
# 移除所有 OpenClaw 相关的二进制文件、配置文件和工作区
x openclaw --uninstall
```

## 交付验证清单
- [ ] 确认用户打算执行安装或配置操作。
- [ ] 确认在设置过程中是否需要特定的集成（如 `qywx`）。
- [ ] 如果存在连接问题，检查网关服务的状态。
