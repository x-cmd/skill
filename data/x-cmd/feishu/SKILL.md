---
name: feishu
description: >
  Send messages to Feishu (Lark) groups using bot webhooks, supporting rich text and cards.
  Core Scenario: When the user needs to send automated notifications or rich text messages to a Feishu group.
license: MIT
---

# feishu - Feishu (Lark) Bot Utility

The `feishu` module enables users to send text and interactive rich text messages to Feishu groups via bot webhooks.

## When to Activate
- When sending automated status updates to a Feishu group.
- When creating rich text notifications with clickable links and titles.
- When integrating Feishu alerts into terminal-based automation scripts.

## Core Principles & Rules
- **Webhook Required**: Configure the `webhook` url via `init` or `--cfg`.
- **Rich Text Support**: Use `--richtext` to include titles, links, and structured text.

## Patterns & Examples

### Send Simple Text
```bash
# Send a basic text message to Feishu
x feishu send --text "New version released"
```

### Send Rich Text
```bash
# Send a message with a title and an embedded link
x feishu send --richtext --title "Project Update" --text "Check details here:" --url "website" "https://x-cmd.com"
```

## Checklist
- [ ] Verify the Feishu webhook URL is set up.
- [ ] Confirm if rich text formatting is required for the message.
