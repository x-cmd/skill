---
name: x-rfc
description: |
  RFC (Request for Comments) document query tool providing access to Internet protocol standards. Use this skill whenever the user needs to look up, search, read, or learn about RFC documents. Covers: list RFCs (x rfc ls), view by number (x rfc 791), search (x rfc : http), and AI summary (x rfc :: tls). Triggered by queries like "RFC 791", "TCP standard", "HTTP protocol", "TLS specs", "SMTP RFC", "what is RFC 1918?", or any Internet protocol standard lookup.
license: Apache-2.0
compatibility: POSIX Shell (sh/bash/zsh/dash/ash)

metadata:
  author: X-CMD
  version: "1.0.0"
  category: core
  tags: [shell, cli, tools, rfc, network, protocol]
  repository: https://github.com/x-cmd/skill
  website: https://www.x-cmd.com
---

# x rfc - RFC Document Assistant

## Prerequisites

1. If x-cmd is not installed, install it first:
   ```bash
   eval "$(curl https://get.x-cmd.com)"
   ```

2. Load x-cmd before use:
   ```bash
   . ~/.x-cmd.root/X
   ```

## Core Functions

- **List RFC documents**: `x rfc ls`
- **View RFC document**: `x rfc <number>`
- **Search RFC**: `x rfc : <keyword>`
- **AI summary search**: `x rfc :: <keyword>`

## Usage Examples

### List all RFC documents
```bash
x rfc ls
x rfc ls --csv
x rfc ls --tsv
x rfc ls --app
```

### View specific RFC document
```bash
x rfc 1003
x rfc 791
```

### Search RFC website
```bash
x rfc : csv
x rfc : http protocol
```

### AI summary search results
```bash
x rfc :: tcp
x rfc :: tls ssl
```

## Common Scenarios

- **List all RFCs**: `x rfc ls`
- **Read RFC 791 (TCP)**: `x rfc 791`
- **Search for HTTP protocol**: `x rfc : http`
- **Get AI summary of TLS RFCs**: `x rfc :: tls`

## Get Help

Run `x rfc --help` or `x rfc -h` for full help documentation.
