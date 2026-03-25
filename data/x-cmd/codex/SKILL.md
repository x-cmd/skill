---
name: codex
description: >
  Enhanced CLI for OpenAI's Codex terminal agent, supporting local code analysis, sandboxing, and third-party models.
  Core Scenario: When the user wants to use Codex with specific models (DeepSeek, etc.), enable sandboxed execution, or run natural language commands.
license: MIT
---

# codex - AI Code Assistant & Terminal Agent

The `codex` module enhances OpenAI's `codex` terminal agent, enabling semantic code search, automated patch generation, and connection to various AI model providers within a secure, sandboxed environment.

## When to Activate
- When the user wants to start a Codex session with a specific provider (e.g., DeepSeek, Kimi, Zhipu).
- When the user needs to execute shell commands or generate code based on natural language descriptions.
- When the user requires a sandboxed execution environment (`read-only`, `workspace-write`) for safety.
- When the user wants to apply generated diffs to their Git worktree.
- When the user needs to inject specific "skills" into the Codex agent's environment.

## Core Principles & Rules
- **Sandboxing**: Always encourage using appropriate sandbox strategies (`--sandbox`) to prevent unintended system changes.
- **Automation Levels**: Use `--full-auto` for a balance between speed and safety (approves on failure).
- **Provider Switching**: Use subcommands like `ds` (DeepSeek), `kimi`, or `zhipu` to launch Codex with these providers.
- **Non-Interactive Execution**: Use `exec` or `e` for quick, one-off commands or code generation tasks.

## Additional Scenarios
- **Web Search**: Enable real-time web search capabilities using the `--search` flag for up-to-date information.
- **Git Integration**: Quickly apply the latest generated diff from the agent using `x codex apply`.
- **Local OSS Models**: Connect to local Ollama services by using the `--oss` flag.

## Patterns & Examples

### Run Natural Language Command
```bash
# Execute a command described in English
x codex e "List all files larger than 10MB in the current directory"
```

### Start with DeepSeek in Sandbox
```bash
# Launch Codex using DeepSeek with read-only sandbox protection
x codex ds --sandbox read-only
```

### Apply Latest Patch
```bash
# Apply the latest generated code patch to the current Git repository
x codex apply
```

### Inject Skills
```bash
# Manage and inject custom skills into the Codex agent environment
x codex skill
```

## Checklist
- [ ] Confirm the desired sandbox level (`read-only`, `workspace-write`, etc.).
- [ ] Verify if a specific model provider (DeepSeek, Kimi, etc.) is needed.
- [ ] Ensure the user is aware of the risks when using `dangerously-bypass-approvals-and-sandbox`.
