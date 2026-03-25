---
name: docker
description: >
  Enhanced Docker management tool with support for mirror acceleration, structured data exports, and container analysis.
  Core Scenario: When the user needs to manage Docker containers and images with optimized performance or export states.
license: MIT
---

# docker - Enhanced Docker Management

The `docker` module provides an enhanced set of tools for Docker, offering mirror acceleration for faster image pulls, structured data exports (JSON/CSV), and interactive TUI explorers.

## When to Activate
- When pulling images from Docker Hub or other registries (automatically uses mirrors).
- When exporting container, image, or volume lists to JSON/CSV for scripting.
- When performing interactive analysis of running containers using the `app` or `fz` modes.
- When managing Docker configurations and mirror registry settings.

## Core Principles & Rules
- **Acceleration**: Transparently applies mirrors to speed up image pulls in specific regions.
- **Structured Data**: Prioritize `--json` or `--csv` for any subcommand when results are for automation.
- **TUI Mode**: Use `ps --app` or `fz` for a visual Docker dashboard experience.

## Patterns & Examples

### Interactive Dashboard
```bash
# Open an interactive TUI to manage running containers
x docker ps --app
```

### Mirror Configuration
```bash
# Set up Docker to use an optimized mirror registry
x docker mirror use ustc
```

### Export Image List
```bash
# Get all local images as a JSON array
x docker images --json
```

## Checklist
- [ ] Confirm if mirror acceleration is needed for image tasks.
- [ ] Verify the desired output format (Human-readable, JSON, CSV).
- [ ] Ensure Docker daemon is running on the host.
