---
name: starship
description: >
  Starship 提示符的增强接口，具有一键主题切换和交互式预览功能。
  核心场景：当用户需要配置、预览或在不同的 Starship 提示符主题之间切换时。
license: MIT
---

# starship - 增强型 Starship 提示符管理

`starship` 模块为 Starship 提示符提供了一个强大的包装器，通过交互式 FZF 预览和简单的切换命令，简化了主题发现和配置。

## 激活时机
- 当用户想要交互式地预览不同的 Starship 主题时。
- 当在预设或自定义的 Starship 配置之间进行切换时。
- 当管理 Starship 环境变量和提示符功能时。

## 核心原则与规范
- **交互式 FZF**: 默认行为启动所有可用主题的可搜索预览。
- **配置覆盖**: 使用此模块将覆盖 `STARSHIP_CONFIG` 环境变量。
- **会话尝试**: 使用 `try` 在当前会话中测试主题（仅限 bash/zsh）。

## 实战示例

### 交互式预览
```bash
# 交互式浏览并预览 Starship 主题
x starship
```

### 应用主题
```bash
# 永久设置一个特定的主题
x starship use gruvbox-rainbow
```

### 当前配置
```bash
# 查看当前激活主题的详情
x starship current
```

## 交付验证清单
- [ ] 确认主题是应该全局应用还是仅用于当前会话。
- [ ] 验证 Starship 是否已在用户的 Shell 中正确初始化。
