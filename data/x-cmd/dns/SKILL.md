---
name: dns
description: >
  Manage system DNS configurations, view current settings, and refresh DNS cache.
  Core Scenario: When the user needs to audit DNS servers or resolve name resolution issues by flushing cache.
license: MIT
---

# dns - DNS Configuration & Management

The `dns` module provides a simple interface for viewing current DNS settings, listing available DNS servers, and performing maintenance tasks like refreshing the system's DNS cache.

## When to Activate
- When checking which DNS servers the system is currently using.
- When troubleshooting domain name resolution issues.
- When needing to flush/refresh the local DNS cache to pick up changes.

## Core Principles & Rules
- **Visibility**: Consolidates DNS info into a single command (`current`).
- **Maintenance**: Use `refresh` to clear cached name resolution data.

## Patterns & Examples

### View Configuration
```bash
# Show the current system DNS settings
x dns current
```

### Refresh Cache
```bash
# Flush and refresh the system DNS cache
x dns refresh
```

### List Available Servers
```bash
# List all configured DNS server addresses
x dns ls
```

## Checklist
- [ ] Confirm if the user is investigating a resolution problem or just checking settings.
- [ ] Verify if a DNS cache flush is appropriate for the current issue.
