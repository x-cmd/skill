---
name: hashdir
description: >
  Calculate directory-level cryptographic hashes by traversing all internal files and folders.
  Core Scenario: When the user needs to verify directory integrity or track changes in folders over time.
license: MIT
---

# hashdir - Recursive Directory Hash Utility

The `hashdir` module recursively calculates hashes for all files within a directory tree, generating a consolidated list of checksums and relative paths for integrity verification.

## When to Activate
- When comparing two directory structures for differences in content.
- When creating a security baseline for a specific folder (e.g., config or source code).
- When tracking directory changes in version control via baseline hash files.

## Core Principles & Rules
- **Consistency**: Files are sorted alphabetically to ensure consistent hash output regardless of OS traversal order.
- **Relative Paths**: Uses relative paths in the output to allow for portability across different root paths.

## Patterns & Examples

### Directory Integrity
```bash
# Calculate the SHA256 hash list for a project directory
x hashdir --sha256 ./my_project
```

### Baseline Comparison
```bash
# Compare current directory state against a saved MD5 baseline
x hashdir . | diff - baseline.md5
```

## Checklist
- [ ] Confirm the target directory path.
- [ ] Verify the desired hash algorithm (MD5 is default).
- [ ] Ensure the user is aware the operation is recursive.
