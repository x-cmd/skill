---
name: telegram
description: >
  Send messages and files to Telegram groups or channels using bot tokens.
  Core Scenario: When the user needs to automate alerts or share files with Telegram chats via bot.
license: MIT
---

# telegram - Telegram Bot Utility

The `telegram` module allows users to send text and images to Telegram groups, channels, or individual users via a bot token. It supports managing tokens, proxies, and target chat IDs.

## When to Activate
- When automating notifications to a Telegram group or channel.
- When sharing local images or documents with Telegram users from the terminal.
- When managing multiple Telegram bot tokens and default chat destinations.

## Core Principles & Rules
- **Token Required**: Use `init` or `--cfg` to set the bot token.
- **Targeting**: Specify chat IDs or usernames using the `--chat` flag.
- **Proxy Support**: Allows configuring network proxies for restricted environments.

## Patterns & Examples

### Send Text to Group
```bash
# Send a notification to a specific chat ID
x telegram send --text --chat "-967810017" "x-cmd release success"
```

### Send Image to Channel
```bash
# Upload and send an image to a public channel
x telegram send --image --chat "@my_channel" --file_path ./a.png
```

## Checklist
- [ ] Ensure the Telegram bot token is correctly set.
- [ ] Verify the target chat ID or username.
- [ ] Confirm if a proxy is needed for connectivity.
