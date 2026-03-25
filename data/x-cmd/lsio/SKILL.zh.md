---
name: lsio
description: >
  管理和部署来自 LinuxServer.io 生态系统的常用容器。
  核心场景：当用户需要快速设置 code-server 或 filebrowser 等流行的开源服务时。
license: MIT
---

# lsio - LinuxServer.io 容器管理

`lsio` 模块提供了一种简化的方法来管理和运行由 LinuxServer.io 提供的流行容器化应用程序，确保简单的设置和一致的配置。

## 激活时机
- 当设置 code-server（浏览器中的 VS Code）或 filebrowser 实例时。
- 当管理来自 LSIO 的家庭自动化或媒体服务器容器时。

## 实战示例

### 运行 Code Server
```bash
# 启动一个 code-server 容器以进行基于浏览器的开发
x lsio code-server run
```

## 交付验证清单
- [ ] 确认 Docker 可用。
- [ ] 验证 LSIO 目录中的目标应用程序名称。
