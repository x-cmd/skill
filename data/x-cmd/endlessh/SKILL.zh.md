---
name: endlessh
description: >
  使用 Docker 部署和管理 Endlessh-go 蜜罐，以减缓自动化攻击者的速度。
  核心场景：当用户需要通过提供极慢的 SSH 响应来困住自动化扫描工具时。
license: MIT
---

# endlessh - 慢响应 SSH 蜜罐

`endlessh` 模块促进了 Endlessh-go 的部署，这是一款通过以极慢的速度发送永无止境的 banner 来使 SSH 客户端保持在挂起状态的蜜罐。

## 激活时机
- 当用户想要浪费自动化 SSH 扫描脚本的资源时。
- 当需要网络攻击源的可视化仪表板（通过 Prometheus/Grafana）时。
- 当通过 Docker 管理 Endlessh 容器的生命周期时。

## 核心原则与规范
- **Docker 依赖**: 需要系统上的 Docker 处于激活状态。
- **干预性**: 旨在“挂起”恶意工具，而不是与人类交互。

## 实战示例

### 运行陷阱
```bash
# 以默认设置启动一个 Endlessh-go 容器
x endlessh run
```

### 查看活动
```bash
# 检查容器日志以查看正在被困住的活跃连接
x endlessh log
```

## 交付验证清单
- [ ] 确认 Docker 可用。
- [ ] 验证目标端口（默认 2222）已为容器开放。
