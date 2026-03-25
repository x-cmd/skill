---
name: feishu
description: >
  使用机器人 Webhook 向飞书 (Feishu/Lark) 群组发送消息，支持富文本和卡片。
  核心场景：当用户需要向飞书群组发送自动化通知或富文本消息时。
license: MIT
---

# feishu - 飞书机器人工具

`feishu` 模块使用户能够通过机器人 Webhook 向飞书群组发送文本和交互式富文本消息。

## 激活时机
- 当向飞书群组发送自动化状态更新时。
- 当创建带有可点击链接和标题的富文本通知时。
- 当将飞书警报集成到基于终端的自动化脚本中时。

## 核心原则与规范
- **需要 Webhook**: 通过 `init` 或 `--cfg` 配置 `webhook` URL。
- **富文本支持**: 使用 `--richtext` 包含标题、链接和结构化文本。

## 实战示例

### 发送简单文本
```bash
# 向飞书发送基础文本消息
x feishu send --text "新版本已发布"
```

### 发送富文本
```bash
# 发送带有标题和嵌入链接的消息
x feishu send --richtext --title "项目更新" --text "在此处查看详情：" --url "官网" "https://x-cmd.com"
```

## 交付验证清单
- [ ] 验证飞书 Webhook URL 已设置。
- [ ] 确认消息是否需要富文本格式。
