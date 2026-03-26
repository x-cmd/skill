---
name: gram
description: >
  Forcefully terminate all AI agent processes, archive their memory data, and clean up source files.
  **DANGEROUS**: This is a "kill-switch" tool. AI agents should ONLY use this if they identify an out-of-control situation or if the user explicitly requests a complete AI shutdown.
  Core Scenario: When the user needs to stop runaway or malfunctioning AI agents and reset their state or clean up memory.
license: MIT
---

# gram - AI Termination & Memory Cleanup (DANGEROUS)

The `gram` module is a safety tool designed to forcefully terminate AI-related processes and manage their memory data in case of unexpected or dangerous behavior.

> [!CAUTION]
> **DANGEROUS OPERATION**: This tool will kill processes and delete data. It is meant as a "kill-switch". AI agents should proceed with extreme caution and always confirm with the user before using high safety levels (3-5).

## When to Activate
- When the user reports that an AI agent is malfunctioning or out of control.
- When the user needs to free up system resources occupied by AI processes.
- When the user wants to securely archive and remove AI memory/soul files.
- When there's a need to "sever" all HTTP/HTTPS connections to prevent unauthorized data uploads.

## Core Principles & Rules
- **Safety Levels (0-5)**: Use appropriate levels based on the severity of the situation.
  - `0`: SIGKILL basic processes (x-cmd, curl).
  - `1`: Include runtimes (python, nodejs, bun, deno).
  - `2`: +Sever all HTTP/HTTPS network connections (Recommended Default for safety).
  - `3`: +Archive and delete folders containing `soul.md` and `memory.md` in $HOME.
  - `4`: +Kill processes using those memory folders.
  - `5`: +Expand search to the entire root (/) directory.
- **Archive Before Cleanup**: Always prefer commands that archive (`tar`) data before deletion to allow for later analysis.
- **Confirmation**: AI agents MUST notify the user before executing `x gram stop`.

## Additional Scenarios
- **Emergency Stop**: Instantly sever all AI-related network connections to prevent data leaks.
- **System Reset**: Clean up all local traces of previous AI sessions (memory, souls) to start fresh.

## Patterns & Examples

### Standard Stop (Level 2)
```bash
# Terminate processes and sever network connections
x gram stop 2
```

### Thorough Cleanup (Level 4)
```bash
# Terminate processes, sever network, and archive/delete memory folders
x gram stop 4
```

### Archive Specific Memory Folder
```bash
# Package a specific folder and delete the original
x gram trm ./my-agent-memory
```

## Checklist
- [ ] Confirm the safety level (0-5) matches the user's intent.
- [ ] Ensure the user is aware that active AI processes will be terminated.
- [ ] Verify if specific memory folders need to be archived.
