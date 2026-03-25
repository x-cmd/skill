---
name: kimi
description: >
  Enhancement module for kimi-cli, integrating AI programming capabilities into the terminal workflow.
  Core Scenario: When the user wants to launch the Kimi Code agent for coding assistance, upgrades, or session management.
license: MIT
---

# kimi - Kimi Code CLI Enhancement

The `kimi` module enhances the `kimi-cli` agent, providing a seamless terminal experience for AI-driven coding, session management, and workspace integration.

## When to Activate
- When the user wants to launch the Kimi Code interactive session.
- When the user needs to upgrade or manage the `kimi-cli` tool.
- When integrating Model Context Protocol (MCP) servers with Kimi.
- When continuing or forking previous AI coding sessions.

## Core Principles & Rules
- **Yolo Mode**: Use the `-y` or `--yolo` flag for automatic tool approval if requested.
- **Thinking Mode**: Enable or disable deep thinking using `--thinking` or `--no-thinking`.
- **Environment Management**: Use `--install` or `--upgrade` to ensure the tool is up to date.

## Additional Scenarios
- **TUI Mode**: Run the interactive terminal UI using `x kimi term`.
- **Web Interface**: Launch the Kimi web UI via `x kimi web`.

## Patterns & Examples

### Launch Kimi Code
```bash
# Start an interactive Kimi Code session in the current directory
x kimi
```

### Auto-Approve Commands
```bash
# Run Kimi with automatic approval for all actions (YOLO mode)
x kimi --yolo
```

### Continue Last Session
```bash
# Resume the most recent conversation in the current workspace
x kimi --continue
```

## Checklist
- [ ] Ensure `kimi-cli` is installed; run `x kimi --install` if necessary.
- [ ] Confirm if the user wants auto-approval (YOLO) enabled.
- [ ] Check if MCP configurations need loading.
