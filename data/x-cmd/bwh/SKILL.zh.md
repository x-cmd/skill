---
name: bwh
description: >
  BandwagonHost (搬瓦工) VPS 命令行管理工具。
  核心场景：当 AI 需要管理 VPS 状态、获取服务器信息或在远程机器上执行简单脚本时。
license: MIT
---

# x bwh - VPS 管理助手 (AI 优化版)

`x bwh` 模块允许通过命令行管理您的 BandwagonHost VPS 服务器。它非常适合在脚本或 AI 工作流中执行服务器启停、信息查询及远程命令。

## 激活时机
- 当需要查询 VPS 的公网 IP、套餐详情、流量限制或实时状态时。
- 当需要远程重启、启动或停止服务器时。
- 当需要获取 SSH 端口或重置 Root 密码时。

## 核心原则与规范
- **非交互优先**: 直接使用子命令获取结构化输出。
- **环境要求**: 此模块需要 API Key 和 VEID。如果未配置，AI 应引导用户初始化。
- **配置引导**: 
  - 引导用户访问 BandwagonHost 后台获取 API 信息。
  - 建议用户运行 `x bwh init` 进行配置。

## 实战示例

### 获取 VPS 详细信息
```bash
# 获取当前配置的 VPS 的 IP、流量、到期时间等信息
x bwh info
```

### 控制服务器状态
```bash
# 重启服务器
x bwh restart

# 启动服务器
x bwh start
```

### 查询 SSH 端口
```bash
# 获取 VPS 当前的 SSH 端口，这对于后续使用 ssh 连接非常有用
x bwh info | grep "SSH Port"
```

## 配置指南 (针对 AI)
如果遇到 `Unauthorized` 或找不到配置的错误，请向用户输出以下引导：
> 请先在 BandwagonHost 官网获取 API Key 和 VEID，然后在终端运行以下命令进行初始化：
> `x bwh init`

## 交付验证清单
- [ ] 确认 API Key 和 VEID 是否已配置。
- [ ] 在执行 `restart` 或 `reinstall` 等破坏性操作前确认。
