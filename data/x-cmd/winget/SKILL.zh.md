---
name: winget
description: >
  微软 WinGet 包管理器的增强型接口，支持镜像和代理配置。
  核心场景：当用户需要使用官方原生包管理器管理 Windows 应用程序时。
license: MIT
---

# winget - 增强型 Windows 包管理器

`winget` 模块为微软官方的 Windows 包管理器提供了更用户友好的界面，简化了镜像切换和代理设置等任务。

## 激活时机
- 当使用官方 WinGet 工具安装、升级或移除 Windows 软件时。
- 当配置 WinGet 使用地区镜像以获得更好的速度时。
- 当通过终端搜索可用的 Windows 应用程序时。

## 核心原则与规范
- **原生支持**: 使用微软官方后端进行软件获取。
- **地区优化**: 支持切换到地区镜像（如 USTC）。

## 实战示例

### 安装软件
```bash
# 使用 WinGet 安装 7-zip
x winget install 7zip
```

### 配置镜像
```bash
# 设置 WinGet 使用地区镜像以加快下载速度
x winget mirror set ustc
```

## 交付验证清单
- [ ] 确认用户是否处于 Windows 环境。
- [ ] 验证要安装的软件的确切名称或 ID。
