---
name: webtop
description: >
  使用 LinuxServer.io 容器快速部署和管理浏览器中的桌面环境。
  核心场景：当用户需要一个可通过网页浏览器访问的完整桌面环境以执行隔离任务时。
license: MIT
---

# webtop - 基于浏览器的桌面部署

`webtop` 模块促进了 LinuxServer.io 的 Webtop 容器部署，为用户提供了一个完整的 Linux 桌面环境（XFCE, KDE 等），可以直接从任何网页浏览器访问。

## 激活时机
- 当用户需要一个带有 GUI 的隔离 Linux 环境时。
- 当执行需要桌面环境但希望避免本地安装的测试或任务时。
- 当通过 Docker 管理 Webtop 容器的生命周期时。

## 核心原则与规范
- **Docker 依赖**: 需要安装并激活 Docker。
- **版本支持**: 支持多种桌面环境（OS 变体），如带有各种 UI 类型的 Alpine, Ubuntu, Fedora。

## 实战示例

### 默认部署
```bash
# 启动默认的 Alpine XFCE 桌面环境
x webtop run
```

### 指定操作系统/UI
```bash
# 在浏览器中运行基于 Ubuntu 的 KDE 桌面
x webtop run --os ubuntu --ui kde
```

## 交付验证清单
- [ ] 确保 Docker 正在运行。
- [ ] 验证所需的操作系统和 UI 组合。
- [ ] 检查浏览器端口是否可访问。
