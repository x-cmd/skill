---
name: discord
description: >
  Send messages to Discord channels using bot webhooks.
  Core Scenario: When the user needs to automate notifications or send alerts to a Discord group.
license: MIT
---

# discord - Discord Bot Utility

The `discord` module enables users to send text messages to Discord channels through bot webhooks, making it easy to integrate terminal alerts into Discord workflows.

## When to Activate
- When sending automated alerts or notifications from terminal scripts to Discord.
- When managing Discord bot webhooks for different channels.

## Core Principles & Rules
- **Webhook Required**: Remind users to configure the `webhook` url via `init` or `--cfg`.
- **Simplicity**: Currently focused on straightforward text message delivery.

## Patterns & Examples

### Send Message
```bash
# Send a text notification to the configured Discord channel
x discord send "Server backup completed successfully"
```

## Checklist
- [ ] Ensure the Discord webhook URL is configured.
- [ ] Verify the message content is appropriate for the target channel.
