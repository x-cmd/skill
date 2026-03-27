---
name: x-zuz
description: >
  Unified compression and decompression tool.
  Core Scenario: When AI needs to handle various archive formats (zip, tar, gz, 7z, rar, zst, xz, bz2) in the terminal. x-cmd provides concise aliases and zero-dependency installation.
license: Apache-2.0
---

# x-zuz - Unified Archive/Compression (AI Optimized)

`x-zuz` is a powerful archive processing module. Its core advantages are a **unified interface** and **zero-dependency auto-installation** (it automatically handles missing backends like 7zip or zstd).

## Core Aliases
- `x z`: Alias for compression (zuz compress)
- `x uz`: Alias for decompression (zuz decompress)

## When to Activate
- When multiple files or directories need to be packed into a specific format.
- When extracting any common archive format (zip, tar.gz, 7z, etc.).
- When listing the contents of an archive without extracting it.
- When extracting and immediately deleting the source file to save space.

## Core Principles & Rules
- **Non-interactive First**: Avoid interactive UIs; use command parameters directly.
- **Universal Interface**: No need to remember different extraction commands (like `tar -zxvf` or `unzip`); use `x uz` for everything.
- **Environment Isolation**: Automatically downloads required backends, ensuring operation in any minimal environment.

## Patterns & Examples

### Quick Compression
```bash
# Compress the 'src' directory into output.tar.gz
x z output.tar.gz src/

# Compress multiple files into a zip
x z archive.zip file1.txt file2.txt
```

### Quick Extraction
```bash
# Extract to the current directory
x uz archive.zip

# Extract to a specific directory
x uz backup.tar.xz ./target_dir/
```

### View Archive Contents (Non-interactive)
```bash
# List files inside an archive
x zuz ls archive.7z
```

### Extract and Delete Source (Cleanup Mode)
```bash
# Ideal for handling temporary downloads
x uzr data.zip
```

## Checklist
- [ ] Prioritize using aliases `x z` or `x uz`.
- [ ] Confirm the target path exists.
- [ ] Consider using `x uzr` for automatic cleanup.
