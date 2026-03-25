---
name: id
description: >
  Enhanced user identity information with colorful and structured output.
  Core Scenario: When the user needs to check UID, GID, and group memberships in a highly readable format.
license: MIT
---

# id - Enhanced User Identity

The `id` module provides a more readable and colorful alternative to the standard `id` command, showing user and group IDs along with memberships in a structured key-value format.

## When to Activate
- When the user wants to view their own or another user's identity info (UID, GID).
- When a script needs to retrieve just the numeric UID or GID.
- When performing manual verification of group memberships.

## Core Principles & Rules
- **Readability**: Automatically provides colorful output in TTY sessions.
- **Script-Friendly**: Use `uid_` or `gid_` subcommands to store results directly in variables for automation.
- **Native Fallback**: Reverts to standard behavior for unknown options.

## Patterns & Examples

### View Identity
```bash
# Show colorful identity info for the current user
x id
```

### Check Specific User
```bash
# View identity details for 'root'
x id root
```

### Get Numeric UID
```bash
# Output only the numeric User ID
x id uid
```

## Checklist
- [ ] Confirm if the user is checking their own ID or another user's.
- [ ] Verify if only the numeric ID is required for a script.
