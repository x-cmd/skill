---
name: bwh
description: >
  BandwagonHost VPS 服务器的命令行管理器，提供生命周期管理和服务器诊断。
  核心场景：当用户需要启动、停止、重启或备份其 BandwagonHost VPS 时。
license: MIT
---

# bwh - BandwagonHost VPS 管理工具

`bwh` 模块允许用户直接从命令行管理其 BandwagonHost VPS 服务器。它支持查看信息、电源管理等常用操作，以及快照和重装等高级功能。

## 激活时机
- 当用户想要检查其 VPS 的状态或详情时。
- 当对 VPS 执行电源操作（启动、停止、重启）时。
- 当管理 SSH 密钥或执行远程 Shell 命令时。
- 当执行高级维护（如创建备份、快照或 OS 重装）时。

## 核心原则与规范
- **配置管理**: 提醒用户通过 `cfg` 或 `current` 配置其 VPS API 详情。
- **谨慎操作**: 应格外小心地处理 `kill`、`reinstall` 和 `resetrootpassword` 等子命令。

## 实战示例

### VPS 状态
```bash
# 查看当前 VPS 的详细信息
x bwh info
```

### 电源管理
```bash
# 重启当前激活的 VPS 实例
x bwh restart
```

### 远程 Shell
```bash
# 通过管理器在 VPS 上执行命令
x bwh sh "uptime"
```

## 交付验证清单
- [ ] 验证目标 VPS 配置是否已激活。
- [ ] 确认用户是否打算执行破坏性操作（重装/强杀）。
- [ ] 确保 API 凭据已正确设置。
