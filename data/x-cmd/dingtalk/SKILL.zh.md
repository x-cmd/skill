---
name: dingtalk
description: >
  使用机器人 Webhook 向钉钉群组发送消息，支持 Markdown 和交互式卡片。
  核心场景：当用户需要向钉钉群组发送自动化通知或交互式消息时。
license: MIT
---

# dingtalk - 钉钉机器人工具

`dingtalk` 模块允许用户通过机器人 Webhook 向钉钉群组发送文本、Markdown 和富文本卡片消息。

## 激活时机
- 当向钉钉群组发送自动化的构建或状态通知时。
- 当使用 Markdown 进行格式化警报（标题、列表、加粗文本）时。
- 当创建带有链接和图片的交互式操作卡片时。

## 核心原则与规范
- **需要 Webhook**: 通过 `init` 或 `--cfg` 设置机器人 `webhook` URL。
- **Markdown 支持**: 使用 `--markdown` 发送带样式的消息。
- **卡片消息**: 使用 `--richtext` 和 `--card` 实现复杂的交互式布局。

## 实战示例

### 发送 Markdown
```bash
# 向钉钉发送带样式的 Markdown 通知
x dingtalk send --markdown --title "发布说明" "## 版本 v1.0.0\n- Bug 修复\n- 新功能"
```

### 发送操作卡片
```bash
# 发送带有链接和图片的富文本卡片
x dingtalk send --richtext --card "查看详情" "https://x-cmd.com" "https://example.com/image.png"
```

## 交付验证清单
- [ ] 确保钉钉 Webhook URL 已配置。
- [ ] 验证消息格式（文本/Markdown/卡片）是否适合内容。
