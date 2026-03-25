---
name: docker
description: >
  增强型 Docker 管理工具，支持镜像加速、结构化数据导出和容器分析。
  核心场景：当用户需要以优化后的性能管理 Docker 容器和镜像，或导出状态时。
license: MIT
---

# docker - 增强型 Docker 管理

`docker` 模块为 Docker 提供了一套增强工具，包括用于加速镜像拉取的镜像加速、结构化数据导出（JSON/CSV）以及交互式 TUI 浏览器。

## 激活时机
- 当从 Docker Hub 或其他注册表拉取镜像时（自动使用镜像加速）。
- 当为脚本编写将容器、镜像或卷列表导出为 JSON/CSV 时。
- 当使用 `app` 或 `fz` 模式对运行中的容器执行交互式分析时。
- 当管理 Docker 配置和镜像加速设置时。

## 核心原则与规范
- **加速**: 透明地应用镜像源，以在特定地区加速镜像拉取。
- **结构化数据**: 当结果用于自动化时，为任何子命令优先使用 `--json` 或 `--csv`。
- **TUI 模式**: 使用 `ps --app` 或 `fz` 获得视觉化的 Docker 仪表板体验。

## 实战示例

### 交互式仪表板
```bash
# 打开交互式 TUI 来管理运行中的容器
x docker ps --app
```

### 镜像加速配置
```bash
# 设置 Docker 使用优化的镜像源
x docker mirror use ustc
```

### 导出镜像列表
```bash
# 以 JSON 数组形式获取所有本地镜像
x docker images --json
```

## 交付验证清单
- [ ] 确认镜像任务是否需要镜像加速。
- [ ] 验证所需的输出格式（人类可读、JSON, CSV）。
- [ ] 确保宿主机上的 Docker 守护进程正在运行。
