---
name: discord
description: >
  使用机器人 Webhook 向 Discord 频道发送消息。
  核心场景：当用户需要自动化通知或向 Discord 群组发送警报时。
license: MIT
---

# discord - Discord 机器人工具

`discord` 模块使用户能够通过机器人 Webhook 向 Discord 频道发送文本消息，从而轻松地将终端警报集成到 Discord 工作流中。

## 激活时机
- 当从终端脚本向 Discord 发送自动化警报或通知时。
- 当为不同频道管理 Discord 机器人 Webhook 时。

## 核心原则与规范
- **需要 Webhook**: 提醒用户通过 `init` 或 `--cfg` 配置 `webhook` URL。
- **简洁性**: 目前专注于直接的文本消息传递。

## 实战示例

### 发送消息
```bash
# 向配置好的 Discord 频道发送文本通知
x discord send "服务器备份已成功完成"
```

## 交付验证清单
- [ ] 确保 Discord Webhook URL 已配置。
- [ ] 验证消息内容是否适合目标频道。
