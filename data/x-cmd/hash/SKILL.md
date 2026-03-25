---
name: hash
description: >
  Calculate and verify cryptographic hashes (MD5, SHA256, etc.) for files or piped input.
  Core Scenario: When the user needs to verify file integrity, generate checksums, or compare hashes.
license: MIT
---

# hash - Cryptographic Checksum Utility

The `hash` module provides a unified interface for calculating MD5, SHA1, SHA256, SHA384, and SHA512 hashes. It includes tools for asserting matches and speeding up calculations via cosmo utilities.

## When to Activate
- When verifying the integrity of downloaded files (e.g., ISOs, packages).
- When generating checksums for files to be shared or archived.
- When performing silent hash comparisons in scripts (`match`).
- When checking multiple files simultaneously.

## Core Principles & Rules
- **Multi-Algorithm**: Automatically detects the algorithm when verifying based on hash length.
- **Speedup**: Supports `speedup` subcommand to install optimized hash tools.
- **Script-Ready**: Use `match` for silent success/failure detection in automation.

## Patterns & Examples

### Calculate Hash
```bash
# Get the SHA256 checksum of a file
x hash sha256 myfile.txt
```

### Assert Match
```bash
# Verify a file against an expected hash with visual feedback
x hash assert myfile.txt EXPECTED_HASH_STRING
```

### Batch Processing
```bash
# Calculate MD5 for all .txt files in the directory
x hash md5 *.txt
```

## Checklist
- [ ] Confirm the target file and desired hash algorithm.
- [ ] Verify if an interactive assertion or a silent match is needed.
- [ ] Ensure the expected hash string is correctly provided.
