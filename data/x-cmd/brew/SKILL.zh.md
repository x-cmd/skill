---
name: brew
description: >
  适用于 macOS 和 Linux 的增强型 Homebrew 封装，管理软件包、镜像和代理。
  核心场景：当用户需要在优化下载的情况下安装、管理或配置 Homebrew 软件包时。
license: MIT
---

# brew - 增强型 Homebrew 封装

`brew` 模块通过提供对镜像配置、代理设置和交互式包选择等常见任务的便捷访问，增强了标准的 Homebrew 体验。

## 激活时机
- 在 macOS 或 Linux 上安装、移除或罗列 Homebrew 软件包时。
- 当配置 Homebrew 使用优化的地区镜像（如 TUNA）时。
- 当为受限网络管理 Homebrew 的代理设置时。
- 当执行交互式软件包浏览 (`x brew` 应用) 时。

## 核心原则与规范
- **交互式浏览**: 使用默认的 `x brew` 通过 TUI 选择软件包。
- **镜像支持**: 使用 `mirror` 子命令简化镜像切换。
- **隐私关注**: 通过 `analytics` 子命令轻松禁用 Homebrew 分析。

## 实战示例

### 安装软件包
```bash
# 一次安装多个软件包
x brew install curl wget git
```

### 配置镜像
```bash
# 设置 Homebrew 使用清华大学镜像源
x brew mirror set tuna
```

### 交互式应用
```bash
# 启动交互式 TUI 来管理 Homebrew 软件包
x brew
```

## 交付验证清单
- [ ] 确认要安装或管理的软件包名称。
- [ ] 验证是否需要设置镜像或代理以获得更好的连通性。
- [ ] 检查系统是 macOS 还是 Linux（受支持的平台）。
