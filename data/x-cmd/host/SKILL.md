---
name: host
description: >
  Securely manage /etc/hosts with interactive search, subcommands for adding/removing entries, and automated backups.
  Core Scenario: When the user needs to edit local hostname-to-IP mappings or perform fuzzy searches in their hosts file.
license: MIT
---

# host - Local Host Table Management

The `host` module provides a safe and efficient way to manage the local `/etc/hosts` file. It handles sudo permissions, provides automated backups before edits, and supports interactive fuzzy searching.

## When to Activate
- When the user wants to add or remove local hostname-to-IP mappings.
- When performing a quick lookup of an IP for a local hostname.
- When searching the hosts file interactively using FZF (`fz`).
- When viewing the full contents of the hosts file with pagination.

## Core Principles & Rules
- **Safety**: Automatically creates backups before any modification (`ed` subcommand).
- **Interactive Search**: Use `fz` for a searchable TUI experience.
- **Privilege Handling**: Clearly indicates that editing requires `sudo` access.

## Patterns & Examples

### Add/Edit Mapping
```bash
# Map a local development domain to an IP
x host ed myapp.local=192.168.1.100
```

### Remove Mapping
```bash
# Delete a specific hostname mapping (prefix with -)
x host ed -myapp.local
```

### Search Hosts
```bash
# Interactively search for an entry in the hosts file
x host fz
```

## Checklist
- [ ] Confirm the target hostname and IP address for edits.
- [ ] Verify if the user has sudo permissions for modification.
- [ ] Ensure the operation (add/remove) is clearly understood.
