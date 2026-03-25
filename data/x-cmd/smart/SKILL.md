---
name: smart
description: >
  Enhanced smartctl interface for disk health monitoring, supporting interactive UI and AI report generation.
  Core Scenario: When the user needs to check disk health (SMART), list disk devices, or generate AI-assisted health reports.
license: MIT
---

# smart - Disk Health & Diagnostics

The `smart` module provides an enhanced CLI for `smartctl`, enabling users to monitor disk health, access self-monitoring (SMART) data, and even leverage AI to interpret technical diagnostic results.

## When to Activate
- When performing hardware health checks on local disks.
- When needing an interactive list of disk devices to choose from.
- When using Gemini to generate human-readable reports from technical SMART data.
- When searching for disk health documentation on smartmontools.com.

## Core Principles & Rules
- **Elevated Privileges**: Automatically invokes `sudo` if needed to access raw disk data.
- **AI Integration**: Designed to pipe diagnostic output into `@gemini` for simplified interpretation.
- **FZF Integration**: Use the `--app` mode for an interactive TUI selection.

## Patterns & Examples

### List Disks
```bash
# Show all available disk paths
x smart --ls
```

### Full Diagnostics (Interactive)
```bash
# Open interactive UI to select disk and view health info
x smart
```

### AI-Assisted Reporting
```bash
# Generate a diagnostic report using Gemini for a specific disk
x smart -a /dev/disk0 | @gemini "generate a health report"
```

## Checklist
- [ ] Confirm the target disk path (e.g., `/dev/disk0`).
- [ ] Verify if the user has the necessary permissions (automated sudo available).
- [ ] Ensure AI integration is requested for report generation.
