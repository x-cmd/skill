---
name: lms
description: >
  CLI module for LM Studio, enabling terminal-based chat and local LLM management.
  Core Scenario: When the user wants to interact with locally hosted models in LM Studio via the command line.
license: MIT
---

# lms - LM Studio CLI Enhancement

The `lms` module provides a CLI interface for LM Studio, allowing users to chat with local models and manage configurations directly from the terminal.

## When to Activate
- When the user wants to chat with models running in LM Studio.
- When managing local LM Studio configurations and session defaults.
- When performing terminal-based interaction with local AI services.

## Core Principles & Rules
- **Integration**: Designed to work alongside the LM Studio desktop application.
- **Subcommand Transparency**: Use `--runcmd` to access original `lms` command features if needed.

## Patterns & Examples

### Chat with Local Model
```bash
# Start a chat session with the model active in LM Studio
x lms chat
```

### Initialize Config
```bash
# Set up default parameters for LM Studio interaction
x lms init
```

## Checklist
- [ ] Ensure LM Studio is running and the local server is active.
- [ ] Verify if specific session defaults need to be set via `x lms --cur`.
