---
name: claude
description: >
  Enhanced CLI module for Claude Code, supporting third-party model providers and session management.
  Core Scenario: When the user wants to use Claude Code with alternative models (DeepSeek, etc.), track token usage, or manage MCP servers.
license: MIT
---

# claude - Claude Code Enhancement Module

The `claude` module extends the capabilities of Anthropic's `claude-code`, allowing users to connect to various AI model providers, manage sessions, and monitor token usage through a unified CLI.

## When to Activate
- When the user wants to start a Claude Code session with a specific provider (e.g., DeepSeek, MiniMax, Zhipu).
- When the user needs to configure Claude Code to use a third-party model globally or per project.
- When the user wants to track token consumption and costs for the last 7 days.
- When the user needs to manage Model Context Protocol (MCP) servers.
- When the user wants to clean up or customize Claude Code's status line or author attribution.

## Core Principles & Rules
- **Provider Connection**: Use subcommands like `ds` (DeepSeek), `mm` (MiniMax), `zhipu`, or `or` (OpenRouter) to launch Claude Code with those specific backends.
- **Global vs. Project Config**: Use `use` for global settings and `use --project` for project-specific model overrides.
- **Session Management**: Use `sess` (Interactive FZF App) or `resume` to manage and recover previous conversations.
- **Security**: Be cautious when using the `--dangerously-skip-permissions` flag; it should only be recommended in trusted sandbox environments.

## Additional Scenarios
- **Local LLM**: Connect Claude Code to a local Ollama service using `x claude use other`.
- **Token Analysis**: Run `x claude usage` to get a detailed breakdown of costs and token count.
- **Clean Git History**: Remove the automatic Co-Author signature from Git commits using `x claude attribution rm`.

## Patterns & Examples

### Launch with DeepSeek
```bash
# Start Claude Code using the DeepSeek model
x claude ds
```

### Configure Project Model
```bash
# Set the current project to use Zhipu (GLM) model provider
x claude use --project glm
```

### Track Usage
```bash
# Analyze token usage and costs for the past 7 days
x claude usage
```

### Interactive Session Management
```bash
# Browse and manage local sessions via FZF
x claude sess
```

## Checklist
- [ ] Confirm if the user wants to use a specific model provider (DeepSeek, MiniMax, etc.).
- [ ] Verify if the configuration should be global or project-specific.
- [ ] Ensure the relevant API Key is configured for the chosen provider.
