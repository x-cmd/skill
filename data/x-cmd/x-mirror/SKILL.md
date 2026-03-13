---
name: x-mirror
description: |
  x-mirror is a comprehensive mirror source management tool for various package managers. Use this skill whenever users need to configure, switch, or query package manager download mirrors. This skill handles: setting up and switching mirrors for package managers (pip, npm, brew, apt, go, cargo, gem, etc.), viewing available mirror options, checking current mirror configuration, restoring default official sources, and optimizing download speeds in China regions. Triggered by queries like "set npm mirror", "change pip source", "npm mirror speed", "pip Tsinghua mirror", "brew Chinese mirror", "x mirror apt", "how to speed up npm download", "configure pip mirror", "switch npm registry", or any request related to package manager mirrors, registry settings, or download acceleration. This skill is essential for developers in China or anyone needing to optimize package download speeds.
license: Apache-2.0
compatibility: POSIX Shell (sh/bash/zsh/dash/ash)

metadata:
  author: X-CMD
  version: "1.0.0"
  category: core
  tags: [shell, cli, tools, mirror, registry, apt, brew, npm, pip, cargo, go, gem, package-manager]
  repository: https://github.com/x-cmd/x-mirror
  install_doc: data/install.md
  display_name: Mirror Source Manager
---

# x mirror - Mirror Source Manager

## Prerequisites

1. Load x-cmd before use:
   ```bash
   . ~/.x-cmd.root/X
   ```

2. x-cmd not installed? → [data/install.md](data/install.md)

## Core Functions

- **List mirror sources**: `x mirror <pkgmgr> ls`
- **Set mirror source**: `x mirror <pkgmgr> set <mirror-name>`
- **View current mirror**: `x mirror <pkgmgr> current`
- **Restore default source**: `x mirror <pkgmgr> unset`

## Supported Package Managers

### System Package Managers
| Command | Description |
|---------|-------------|
| apt | Debian/Ubuntu |
| brew | Homebrew |
| pacman | Arch Linux |
| dnf | Fedora/RHEL |
| yum | CentOS/RHEL |
| apk | Alpine Linux |

### Language Package Managers
| Command | Description |
|---------|-------------|
| pip | Python |
| npm | Node.js |
| pnpm | Node.js |
| yarn | Node.js |
| go | Go modules |
| cargo | Rust |
| gem | Ruby |

### Container
| Command | Description |
|---------|-------------|
| docker | Docker registry |

## Usage Examples

### List available mirrors
```bash
x mirror npm ls
x mirror pip ls
x mirror brew ls
```

### Set mirror source
```bash
x mirror npm set npmmirror    # Set npm to use Alibaba Cloud mirror
x mirror pip set tuna         # Set pip to use Tsinghua mirror
x mirror brew set tuna        # Set brew to use Tsinghua mirror
```

### View current mirror
```bash
x mirror npm current
x mirror pip current
```

### Restore default source
```bash
x mirror npm unset
x mirror pip unset
```

## Common Scenarios

- **Check available npm mirrors**: `x mirror npm ls`
- **Set pip to Tsinghua mirror**: `x mirror pip set tuna`
- **Check current brew mirror**: `x mirror brew current`
- **Restore npm to official**: `x mirror npm unset`

## Get Help

Run `x mirror --help` or `x mirror <subcmd> --help` for full documentation.
