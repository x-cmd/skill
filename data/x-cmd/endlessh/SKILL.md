---
name: endlessh
description: >
  Deploy and manage Endlessh-go honeypots using Docker to slow down automated attackers.
  Core Scenario: When the user needs to trap automated scanning tools by providing extremely slow SSH responses.
license: MIT
---

# endlessh - Slow-Response SSH Honeypot

The `endlessh` module facilitates the deployment of Endlessh-go, a honeypot that keeps SSH clients locked in a pending state by sending never-ending banners at a very slow pace.

## When to Activate
- When the user wants to waste the resources of automated SSH scanning scripts.
- When needing a visual dashboard (via Prometheus/Grafana) of network attack sources.
- When managing the lifecycle of Endlessh containers via Docker.

## Core Principles & Rules
- **Docker Dependency**: Requires Docker to be active on the system.
- **Intervention**: Designed to "hang" malicious tools, not to interact with humans.

## Patterns & Examples

### Run Trapper
```bash
# Start an Endlessh-go container with default settings
x endlessh run
```

### View Activity
```bash
# Check the container logs to see active connections being trapped
x endlessh log
```

## Checklist
- [ ] Confirm Docker is available.
- [ ] Verify the target port (default 2222) is open for the container.
