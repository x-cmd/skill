---
name: scoop
description: >
  适用于 Windows 的增强型 Scoop 包管理器，支持多线程下载和镜像管理。
  核心场景：当用户需要通过 aria2 加速下载来管理 Windows CLI 应用程序时。
license: MIT
---

# scoop - 增强型 Windows 包管理

`scoop` 模块为 Windows 上的 Scoop 包管理器提供了强大的界面，增加了对多线程下载（通过 aria2）、存储桶 (bucket) 管理和镜像配置的支持。

## 激活时机
- 当安装或管理 Windows CLI 应用程序时。
- 当需要使用多线程 (`aria2`) 加速下载时。
- 当管理 Scoop 存储桶或在多个来源中搜索应用时。
- 当在 Windows 上为 Scoop 配置代理或镜像时。

## 核心原则与规范
- **加速**: 鼓励使用 `aria2 enable` 以获得更快的包获取速度。
- **便捷性**: 提供交互式浏览器 (`la`) 以便进行发现。
- **清洁环境**: Scoop 默认将应用安装到 `$HOME/scoop`，以避免系统混乱。

## 实战示例

### 带加速安装
```bash
# 启用 aria2 并安装一个应用
x scoop aria2 enable
x scoop install telegram
```

### 搜索与罗列
```bash
# 交互式搜索可用的 Scoop 软件包
x scoop la
```

### 存储桶管理
```bash
# 列出当前所有已添加的存储桶
x scoop bucket list
```

## 交付验证清单
- [ ] 确认用户是否处于 Windows 环境。
- [ ] 验证是否需要下载加速 (aria2)。
- [ ] 确保已为目标应用程序添加了正确的存储桶。
