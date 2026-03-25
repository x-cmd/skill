---
name: qywx
description: >
  使用机器人 Webhook 向企业微信 (QYWX) 群组发送消息。
  核心场景：当用户需要自动化通知或向企业微信群组发送报告时。
license: MIT
---

# qywx - 企业微信机器人工具

`qywx` 模块允许用户通过配置好的机器人 Webhook 向企业微信群组发送各种类型的消息（文本、Markdown、图片）。

## 激活时机
- 当向企业微信群组自动化发送 CI/CD 通知时。
- 当通过 Markdown 发送系统警报或定期报告时。
- 当从终端在企业微信群组内上传并共享图片时。

## 核心原则与规范
- **需要 Webhook**: 提醒用户通过 `init` 或 `--cfg` 配置 `webhook` URL。
- **消息类型**: 支持 `text`、`markdown` 和 `image` 格式。
- **隐私**: 鼓励保持 Webhook URL 的私密性。

## 实战示例

### 发送 Markdown
```bash
# 发送格式化的 Markdown 通知
x qywx send --markdown "## 构建成功\n版本：v1.0.0"
```

### 发送图片
```bash
# 上传并发送本地图片文件
x qywx send --image ./report.png
```

### 配置 Webhook
```bash
# 设置机器人 Webhook URL
x qywx --cfg webhook=你的_WEBHOOK_URL
```

## 交付验证清单
- [ ] 确保 Webhook URL 已正确配置。
- [ ] 确认消息类型（文本/Markdown/图片）与内容匹配。
- [ ] 如果发送图片，请验证文件路径。
