---
name: webtop
description: >
  Rapidly deploy and manage desktop-in-browser environments using LinuxServer.io containers.
  Core Scenario: When the user needs a full desktop environment accessible via web browser for isolated tasks.
license: MIT
---

# webtop - Browser-Based Desktop Deployment

The `webtop` module facilitates the deployment of LinuxServer.io's Webtop containers, providing users with a complete Linux desktop environment (XFCE, KDE, etc.) that can be accessed directly from any web browser.

## When to Activate
- When the user needs an isolated Linux environment with a GUI.
- When performing testing or tasks that require a desktop but want to avoid local installation.
- When managing the lifecycle of Webtop containers via Docker.

## Core Principles & Rules
- **Docker Dependency**: Requires Docker to be installed and active.
- **Flavor Support**: Supports multiple desktop environments (flavors) like Alpine, Ubuntu, Fedora with various UI types.

## Patterns & Examples

### Default Deployment
```bash
# Start a default Alpine XFCE desktop environment
x webtop run
```

### Specific OS/UI
```bash
# Run an Ubuntu-based KDE desktop in the browser
x webtop run --os ubuntu --ui kde
```

## Checklist
- [ ] Ensure Docker is running.
- [ ] Verify the desired OS and UI combination.
- [ ] Check if the browser port is accessible.
