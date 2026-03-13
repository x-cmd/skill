---
name: x-env
description: |
  x-env is x-cmd's environment and package management module for installing and managing third-party software, programming language runtimes, and command-line tools. Use this skill when: installing or managing runtimes (node, python, go, bun, java, rust); installing CLI tools (jq, yq, fzf, himalaya, claude-code); managing multiple software versions (specify versions, switch versions); cleaning up unused packages; viewing installed software paths or dependencies; asking how to install software; temporarily using software in current shell session (try); permanently installing software to user environment (use); upgrading installed software versions; running scripts with specific software versions. This is the core package manager in x-cmd ecosystem.
license: Apache-2.0
compatibility: POSIX Shell (sh/bash/zsh/dash/ash)

metadata:
  author: X-CMD
  version: "1.0.0"
  category: core
  tags: [shell, cli, tools, package-manager, env, runtime]
  repository: https://github.com/x-bash/env
  install_doc: data/install.md
  display_name: Environment & Package Manager
---

# x env - Environment & Package Manager

## Prerequisites

1. Load x-cmd before use:
   ```bash
   . ~/.x-cmd.root/X
   ```

2. x-cmd not installed? → [data/install.md](data/install.md)

## Core Concepts

- **try**: Temporarily use software in current shell session (only valid for current terminal)
- **use**: Permanently install software to user environment (persists after terminal restart)
- **untry**: Cancel temporarily tried software in current session
- **unuse**: Remove installed software from user environment
- **upgrade**: Upgrade installed software versions

## Core Functions

- **List available versions**: `x env la <pkg>`
- **List installed packages**: `x env ls`, `x env ll`
- **Temporary use (try)**: `x env try <pkg>[=<version>]`
- **Permanent install (use)**: `x env use <pkg>[=<version>]`
- **Remove package**: `x env unuse <pkg>`
- **Upgrade package**: `x env upgrade <pkg>`
- **Cleanup packages**: `x env gc <pkg>`
- **Find command path**: `x env which <cmd>`
- **Check dependencies**: `x env depend <pkg>`

## Usage Examples

### Install packages
```bash
x env use node              # Install default version of node
x env use python            # Install default version of python
x env use go=v1.21.0        # Install specific version
x env use node python go    # Install multiple packages
```

### Temporary use (current shell session only)
```bash
x env try node              # Use default node in current session
x env try bun go=v1.17.13   # Use multiple packages with versions
x env untry node            # Cancel temporary use
```

### Version management
```bash
x env use --upgrade --all   # Upgrade all installed packages
x env use --upgrade jq yq   # Upgrade specific packages
x env upgrade node python   # Upgrade packages (shorthand)
```

### List and search
```bash
x env la node               # List all available node versions
x env ls                    # List packages in use
x env ll                    # List all installed packages
x env which node            # Show node installation path
```

### Cleanup
```bash
x env gc jq yq              # Remove specified packages and orphans
```

## Common Scenarios

- **Install node**: `x env use node`
- **Install python**: `x env use python`
- **Install go**: `x env use go`
- **Install jq**: `x env use jq`
- **Install specific version**: `x env use node=v18.12.0`
- **Temporary try**: `x env try node`
- **Upgrade all**: `x env upgrade --all`
- **Find path**: `x env which node`

## Get Help

Run `x env --help` for full help documentation.
