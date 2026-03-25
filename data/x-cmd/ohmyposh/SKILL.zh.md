---
name: ohmyposh
description: >
  oh-my-posh 提示符的增强接口，提供主题下载和交互式预览。
  核心场景：当用户需要探索、安装或在 oh-my-posh 提示符主题之间切换时。
license: MIT
---

# ohmyposh - 增强型 oh-my-posh 管理

`ohmyposh` 模块通过提供主题发现工具、交互式 FZF 预览以及在多种 Shell 中一键应用样式的功能，促进了 oh-my-posh 的使用。

## 激活时机
- 当用户想要通过实时预览浏览 oh-my-posh 主题时。
- 当安装或在 oh-my-posh 样式之间进行切换时。
- 当管理字体或升级 oh-my-posh 工具时。

## 核心原则与规范
- **实时预览**: 默认命令 (`x ohmyposh`) 提供可搜索的、交互式的主题预览。
- **配置管理**: 覆盖 `POSH_THEME` 环境变量以确保一致性。
- **Shell 兼容性**: 尝试功能 (`try`/`untry`) 仅限于 bash 和 zsh。

## 实战示例

### 预览与搜索
```bash
# 启动 oh-my-posh 主题的交互式 FZF 预览
x ohmyposh
```

### 设置全局主题
```bash
# 永久使用 'montys' 主题
x ohmyposh use montys
```

## 交付验证清单
- [ ] 确认用户是否需要为该主题安装特定的字体。
- [ ] 验证 Shell 类型以确认尝试功能的兼容性。
