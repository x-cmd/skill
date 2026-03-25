---
name: pb
description: >
  Cross-platform clipboard tool with support for local and remote (SSH) sessions.
  Core Scenario: When the user needs to copy/paste text via terminal, including across SSH connections using OSC52.
license: MIT
---

# pb - Cross-Platform Clipboard

The `pb` module provides a unified interface for system clipboards across macOS, Windows, Linux, and Android. It automatically detects the environment and even supports SSH-based copying through the OSC52 protocol.

## When to Activate
- When the user wants to copy command output or text to the system clipboard.
- When pasting clipboard content into the terminal or a command.
- When working in a remote SSH session and needing to sync the remote clipboard with the local machine.

## Core Principles & Rules
- **Environment Agnostic**: One command (`x pb`) replaces `pbcopy`, `xclip`, `wl-copy`, and `clip.exe`.
- **Remote Synchronization**: Automatically uses OSC52 in SSH sessions to allow copying from a server to a local client.
- **Input Flexibility**: Supports both piped input and direct arguments for copying.

## Patterns & Examples

### Copy from Pipe
```bash
# Copy the result of a command to clipboard
echo "data to copy" | x pb
```

### Copy Arguments
```bash
# Copy specific text directly
x pb copy "Hello World"
```

### Paste to Terminal
```bash
# Paste current clipboard content
x pb paste
```

## Checklist
- [ ] Confirm if the operation is a copy or a paste.
- [ ] Ensure the system has a clipboard manager installed if on Linux (xclip/wl-copy).
- [ ] Verify SSH client support for OSC52 if working remotely.
