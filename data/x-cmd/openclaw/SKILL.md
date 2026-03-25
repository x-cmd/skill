---
name: openclaw
description: >
  Install and configure the OpenClaw personal AI assistant.
  Core Scenario: When the user needs to deploy, uninstall, or configure their own local AI assistant on their device.
license: MIT
---

# openclaw - Personal AI Assistant Deployment

The `openclaw` module provides a one-click initialization and configuration environment for OpenClaw, a personal AI assistant that can be run on local devices.

## When to Activate
- When the user wants to install OpenClaw and its dependencies.
- When the user needs to set up specific integrations for OpenClaw (e.g., enterprise WeChat).
- When the user wants to manage the OpenClaw gateway service.
- When the user needs to uninstall OpenClaw and clean up its workspace.

## Core Principles & Rules
- **Environment Automation**: Use `--install` to automatically detect and set up the necessary environment and dependencies.
- **Service Management**: Use the `service` subcommand to manage the persistent gateway process.
- **Integration Setup**: Use `--setup` to quickly configure models and chat software integrations.

## Additional Scenarios
- **QYWX Integration**: Rapidly configure OpenClaw to work with Enterprise WeChat for communication.
- **Complete Cleanup**: Ensure all binaries, configs, and workspaces are removed during uninstallation.

## Patterns & Examples

### Install OpenClaw
```bash
# Install OpenClaw and its latest dependencies
x openclaw --install
```

### Setup QYWX Integration
```bash
# Configure OpenClaw to use Enterprise WeChat
x openclaw --setup qywx
```

### Manage Gateway Service
```bash
# Check status, start, or stop the gateway service
x openclaw service status
x openclaw service start
```

### Uninstall OpenClaw
```bash
# Remove all OpenClaw-related files and services
x openclaw --uninstall
```

## Checklist
- [ ] Verify that the user intends to perform an installation or configuration.
- [ ] Confirm if specific integrations (like `qywx`) are required during setup.
- [ ] Check the status of the gateway service if there are connectivity issues.
