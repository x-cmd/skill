---
name: x-install
description: |
  Software installation assistant using x-cmd's install module. Use this skill whenever the user needs to query software installation methods, get installation commands, or understand how to install various software packages. Covers: getting installation commands (x install --cat <software>), listing supported software (x install --ll), and more. Triggered by queries like "how to install node", "docker installation command", "install python", "show me git install command", or any software installation lookup.
license: Apache-2.0
compatibility: POSIX Shell (sh/bash/zsh/dash/ash)

metadata:
  author: X-CMD
  version: "1.0.0"
  category: core
  tags: [shell, cli, tools, install, software, package-manager]
  repository: https://github.com/x-cmd/skill
  install_doc: data/install.md
  display_name: Software Installation Assistant
---

# x install - Software Installation Assistant

## Prerequisites

1. Load x-cmd before use:
   ```bash
   . ~/.x-cmd.root/X
   ```

2. x-cmd not installed? → [data/install.md](data/install.md)

## Core Functions

- **Get installation command**: `x install --cat <software>`
- **List supported software**: `x install --ll`

## Usage Examples

### Get installation command for specific software
```bash
x install --cat git
x install --cat docker
x install --cat node
x install --cat python
```

### List all supported software
```bash
x install --ll
```

## Common Scenarios

- **Get Git installation command**: `x install --cat git`
- **Get Docker installation command**: `x install --cat docker`
- **Get Node.js installation command**: `x install --cat node`
- **Get Python installation command**: `x install --cat python`
- **Get Claude Code installation command**: `x install --cat claude-code`

## Get Help

Run `x install --help` for full help documentation.
