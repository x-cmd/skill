---
name: lsio
description: >
  Manage and deploy common containers from the LinuxServer.io ecosystem.
  Core Scenario: When the user needs to quickly set up popular open-source services like code-server or filebrowser.
license: MIT
---

# lsio - LinuxServer.io Container Management

The `lsio` module provides a simplified way to manage and run popular containerized applications provided by LinuxServer.io, ensuring easy setup and consistent configuration.

## When to Activate
- When setting up a code-server (VS Code in browser) or filebrowser instance.
- When managing home automation or media server containers from LSIO.

## Patterns & Examples

### Run Code Server
```bash
# Start a code-server container for browser-based development
x lsio code-server run
```

## Checklist
- [ ] Confirm Docker is available.
- [ ] Verify the target application name from the LSIO catalog.
