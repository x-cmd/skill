---
name: facl
description: >
  Retrieve and manage File Access Control Lists (ACL) in a more intuitive format.
  Core Scenario: When the user needs to view or set fine-grained file permissions (ACLs) via terminal.
license: MIT
---

# facl - File Access Control List Management

The `facl` module simplifies the interaction with File Access Control Lists (ACL), providing an easier way to view and set complex file permissions beyond standard Unix rwx bits.

## When to Activate
- When the user needs to audit fine-grained permissions on files or directories.
- When setting specific user or group permissions using ACLs.

## Core Principles & Rules
- **Intuitive Display**: Designed to show ACL entries in a clear, readable format.
- **Standard Operations**: Supports both `get` (view) and `set` (modify) subcommands.

## Patterns & Examples

### View ACL
```bash
# Get the ACL for the current directory
x facl
# Get the ACL for a specific file
x facl myfile.txt
```

## Checklist
- [ ] Confirm if the user intends to view or modify ACLs.
- [ ] Verify the target file or directory path.
