---
name: x-zuz
description: |
  Handle archive/compression tasks including compressing, extracting, archiving, packaging, listing contents, and viewing files.
  Supports formats: zip, tar, gz, 7z, rar, zst, xz, bz2 and more.
  You MUST use this skill whenever the user mentions or implies any archive, compression, extraction, packaging, or archive viewing needs.
  Even if the user doesn't explicitly mention x-zuz or specific commands, as long as there's an archive-related request, invoke this skill.

license: Apache-2.0
compatibility: POSIX Shell (sh/bash/zsh/dash/ash)

metadata:
  author: X-CMD
  version: "1.0.0"
  category: core
  tags: [shell, cli, tools, compression, archive]
  repository: https://github.com/x-cmd/skill
  website: https://www.x-cmd.com
  install_doc: data/install.md
  display_name: Archive/Compression Assistant
---

# x zuz - Archive/Compression Assistant

## Prerequisites

1. Load x-cmd before use:
   ```bash
   . ~/.x-cmd.root/X
   ```

2. x-cmd not installed? → [data/install.md](data/install.md)

## Core Functions

- **Compress**: `x z <output> <files...>`
- **Extract**: `x uz <archive> [destination]`
- **List contents**: `x zuz ls <archive>`

## Usage Examples

### Compress files and directories
```bash
x z output.zip file1.txt file2.txt dir/
x z archive.tar.xz myfolder/
x z backup.tar.gz ./src/*
```

### Extract archives
```bash
x uz archive.zip
x uz archive.tar.xz ./target_dir/
x uz file.7z /tmp/output/
```

### List archive contents
```bash
x zuz ls archive.zip
x zuz ls archive.tar.gz
```

## Common Scenarios

- **Backup packaging**: `x z backup.tar.xz ./important_files/`
- **Extract downloaded file**: `x uz downloaded.zip`
- **View archive contents**: `x zuz ls old_backup.tar.gz`

## Get Help

Run `x zuz --help` or `x zuz -h` for full help documentation.
