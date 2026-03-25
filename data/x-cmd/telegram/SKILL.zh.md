---
name: telegram
description: >
  使用机器人令牌向 Telegram 群组或频道发送消息和文件。
  核心场景：当用户需要自动化警报或通过机器人与 Telegram 聊天共享文件时。
license: MIT
---

# telegram - Telegram 机器人工具

`telegram` 模块允许用户通过机器人令牌向 Telegram 群组、频道或个人用户发送文本和图片。它支持管理令牌、代理以及目标聊天 ID。

## 激活时机
- 当向 Telegram 群组或频道发送自动化通知时。
- 当从终端与 Telegram 用户共享本地图片或文档时。
- 当管理多个 Telegram 机器人令牌和默认聊天目标时。

## 核心原则与规范
- **需要令牌**: 使用 `init` 或 `--cfg` 设置机器人令牌 (token)。
- **目标指定**: 使用 `--chat` 标志指定聊天 ID 或用户名。
- **代理支持**: 允许为受限环境配置网络代理。

## 实战示例

### 向群组发送文本
```bash
# 向特定的聊天 ID 发送通知
x telegram send --text --chat "-967810017" "x-cmd 发布成功"
```

### 向频道发送图片
```bash
# 上传并向公开频道发送图片
x telegram send --image --chat "@my_channel" --file_path ./a.png
```

## 交付验证清单
- [ ] 确保 Telegram 机器人令牌已正确设置。
- [ ] 验证目标聊天 ID 或用户名。
- [ ] 确认是否需要代理进行连通。
