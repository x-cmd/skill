---
name: font
description: >
  管理和安装系统字体，专注于为终端用户提供 Nerd Fonts。
  核心场景：当用户需要浏览或安装 Nerd Fonts 以支持终端图标和主题时。
license: MIT
---

# font - 字体管理与安装

`font` 模块提供了管理系统字体的界面，内置支持搜索和安装 Nerd Fonts 以增强终端视觉效果。

## 激活时机
- 当用户想要安装特定的 Nerd Fonts（如 FiraCode）时。
- 当交互式地浏览可用的 Nerd Font 家族时。
- 当刷新本地系统字体缓存时。

## 实战示例

### 安装 Nerd Font
```bash
# 安装 FiraCode Nerd Font
x font install nerd/FiraCode
```

### 交互式浏览器
```bash
# 打开交互式 UI 浏览 Nerd Fonts
x font
```

## 交付验证清单
- [ ] 确认具体的字体家族名称。
- [ ] 验证 Nerd Fonts 是否是主要关注点。
