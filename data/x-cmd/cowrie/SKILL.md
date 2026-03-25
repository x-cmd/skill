---
name: cowrie
description: >
  Rapidly deploy and manage Cowrie honeypot containers using Docker for network defense research.
  Core Scenario: When the user needs to set up an interactive SSH/Telnet honeypot to monitor attacker behavior.
license: MIT
---

# cowrie - Docker-Based Honeypot Deployment

The `cowrie` module simplifies the deployment of the Cowrie honeypot, an interactive tool designed to mimic SSH and Telnet services to log attacker behavior and collect malware.

## When to Activate
- When the user wants to set up a honeypot environment for security research.
- When monitoring unauthorized login attempts and capturing attacker commands.
- When managing the lifecycle (run, stop, restart) of Cowrie Docker containers.

## Core Principles & Rules
- **Docker Dependency**: Requires Docker to be installed and running on the host system.
- **Isolated Testing**: Use the `test` subcommand to verify honeypot connectivity internally.
- **Logging**: Prioritize using the `log` subcommand to analyze captured data.

## Patterns & Examples

### Run Honeypot
```bash
# Start a default Cowrie honeypot container
x cowrie run
```

### Test Connectivity
```bash
# Locally test the SSH connection to the running honeypot (port 2223)
x cowrie test
```

### Check Logs
```bash
# View the live logs from the Cowrie honeypot
x cowrie log
```

## Checklist
- [ ] Ensure Docker is active on the system.
- [ ] Verify if a custom container name is required (`--name`).
- [ ] Confirm if the user is aware this mimics a vulnerable service.
