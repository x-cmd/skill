---
name: qywx
description: >
  Send messages to Enterprise WeChat (QYWX) groups using bot webhooks.
  Core Scenario: When the user needs to automate notifications or send reports to a QYWX group.
license: MIT
---

# qywx - Enterprise WeChat Bot Utility

The `qywx` module allows users to send various types of messages (text, markdown, images) to Enterprise WeChat groups via a configured bot webhook.

## When to Activate
- When automating CI/CD notifications to a QYWX group.
- When sending system alerts or periodic reports via markdown.
- When uploading and sharing images within a QYWX group from the terminal.

## Core Principles & Rules
- **Webhook Required**: Remind users to configure the `webhook` url via `init` or `--cfg`.
- **Message Types**: Supports `text`, `markdown`, and `image` formats.
- **Privacy**: Encourage keeping the webhook URL private.

## Patterns & Examples

### Send Markdown
```bash
# Send a formatted markdown notification
x qywx send --markdown "## Build Success\nVersion: v1.0.0"
```

### Send Image
```bash
# Upload and send a local image file
x qywx send --image ./report.png
```

### Configure Webhook
```bash
# Set up the bot webhook URL
x qywx --cfg webhook=YOUR_WEBHOOK_URL
```

## Checklist
- [ ] Ensure the webhook URL is correctly configured.
- [ ] Confirm the message type (text/markdown/image) matches the content.
- [ ] Verify file paths if sending an image.
