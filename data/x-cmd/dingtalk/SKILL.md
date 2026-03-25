---
name: dingtalk
description: >
  Send messages to DingTalk groups using bot webhooks, supporting Markdown and interactive cards.
  Core Scenario: When the user needs to automate notifications or send interactive messages to a DingTalk group.
license: MIT
---

# dingtalk - DingTalk Bot Utility

The `dingtalk` module allows users to send text, markdown, and rich text card messages to DingTalk groups via bot webhooks.

## When to Activate
- When sending automated build or status notifications to a DingTalk group.
- When using Markdown for formatted alerts (titles, lists, bold text).
- When creating interactive action cards with links and images.

## Core Principles & Rules
- **Webhook Required**: Set the bot `webhook` url via `init` or `--cfg`.
- **Markdown Support**: Use `--markdown` for styled messages.
- **Card Messages**: Use `--richtext` and `--card` for complex interactive layouts.

## Patterns & Examples

### Send Markdown
```bash
# Send a styled markdown notification to DingTalk
x dingtalk send --markdown --title "Release Note" "## Version v1.0.0\n- Bug fixes\n- New features"
```

### Send Action Card
```bash
# Send a rich text card with a link and an image
x dingtalk send --richtext --card "View Details" "https://x-cmd.com" "https://example.com/image.png"
```

## Checklist
- [ ] Ensure the DingTalk webhook URL is configured.
- [ ] Verify the message format (text/markdown/card) suits the content.
