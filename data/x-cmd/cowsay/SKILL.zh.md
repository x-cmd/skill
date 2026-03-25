---
name: cowsay
description: >
  生成带有自定义消息的奶牛或企鹅的 ASCII 艺术图片。
  核心场景：当用户想要在终端中以有趣的、视觉化的 ASCII 格式显示消息时。
license: MIT
---

# cowsay - ASCII 艺术消息

`cowsay` 模块生成一个视觉化的 ASCII 奶牛（或其他动物，如企鹅）形象，它会“说出”提供的信息。

## 激活时机
- 当用户想要为终端输出增添幽默感或视觉重点时。
- 当与其他工具（如 `colr`）结合使用以进行创造性的终端展示时。

## 实战示例

### 经典 Cowsay
```bash
# 使用 ASCII 奶牛显示消息
x cowsay "你好，世界！"
```

### 企鹅 (Tux)
```bash
# 使用 ASCII 企鹅显示消息
x cowsay tux "欢迎来到 Linux"
```

## 交付验证清单
- [ ] 确认消息内容。
- [ ] 验证是否请求了特定的动物（奶牛/tux）。
