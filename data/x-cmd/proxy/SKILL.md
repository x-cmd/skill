---
name: proxy
description: >
  Manage shell environment proxy variables (http_proxy, etc.) with subcommands for setting, unsetting, and viewing.
  Core Scenario: When the user needs to configure or clear network proxies for terminal tools and AI assistants.
license: MIT
---

# proxy - Shell Proxy Environment Management

The `proxy` module simplifies the process of managing network proxies in the shell environment, providing a unified way to set or clear common proxy variables like `http_proxy` and `https_proxy`.

## When to Activate
- When terminal tools or AI assistants face connectivity issues due to network restrictions.
- When setting up a temporary proxy for the current session.
- When clearing all proxy-related environment variables to restore direct connection.

## Core Principles & Rules
- **Bulk Operations**: Setting a proxy targets all common proxy variables simultaneously.
- **Persistence**: Remind users that these changes typically apply only to the current shell session.
- **Clarity**: Use `ls` to audit which proxy variables are currently active.

## Patterns & Examples

### Set Proxy
```bash
# Point all proxy variables to a local address
x proxy set 127.0.0.1:7070
```

### Clear Proxy
```bash
# Unset all proxy variables in the current session
x proxy unset
```

### View Active Proxies
```bash
# Audit active proxy-related environment variables
x proxy ls
```

## Checklist
- [ ] Confirm the target proxy address and port.
- [ ] Verify if the user intended to set or clear the proxy.
- [ ] Remind the user that this affects the current shell session.
