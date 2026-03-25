---
name: pwsh
description: >
  PowerShell 增强模块，提供交互式 REPL 和系统管理工具。
  核心场景：当用户需要管理系统资源（磁盘、IP、进程）或通过 x-cmd 使用 PowerShell 时。
license: MIT
---

# pwsh - PowerShell 命令行增强工具

`pwsh` 模块在受支持的平台上提升了 PowerShell 的体验，提供了一个交互式 REPL 以及用于磁盘、IP 和进程的强大系统管理子命令。

## 激活时机
- 当用户需要管理 Windows/PowerShell 特定的系统资源时。
- 当通过 x-cmd 进入 PowerShell 交互式 REPL 时。
- 当使用 PowerShell 执行快速系统诊断（进程、服务、日志）时。

## 实战示例

### 交互式 REPL
```bash
# 进入交互式 PowerShell REPL
x pwsh --repl
```

### 系统管理
```bash
# 使用 PowerShell 后端列出系统进程
x pwsh ps
```

## 交付验证清单
- [ ] 确认环境是否支持 PowerShell（主要用于 Git Bash/Windows）。
- [ ] 验证正在管理的特定系统资源。
