---
name: crush
description: >
  AI-powered CLI tool by Charm for code analysis, diagnostics, and interactive Q&A.
  Core Scenario: When the user needs intelligent code suggestions, explanations, or fast local development diagnostics.
license: MIT
---

# crush - AI Assistant for Code Analysis

`crush` is an AI-driven tool developed by Charm that provides context-aware suggestions and code explanations, integrating with Language Server Protocol (LSP) for a smooth developer experience.

## When to Activate
- When the user wants interactive AI assistance for code analysis.
- When performing fast diagnostics or seeking intelligent suggestions within a project.
- When running single, non-interactive AI prompts for code explanation.
- When the user needs a debugged or YOLO-approved execution of AI commands.

## Core Principles & Rules
- **API Key Inheritance**: Inherits `OPENAI_API_KEY` and `GEMINI_API_KEY` from `x openai` and `x gemini` configurations.
- **Context Awareness**: Best used within a specific project directory (`--cwd`) to leverage local code context.
- **YOLO Mode**: Use `-y` or `--yolo` to automatically accept permissions for high-speed interaction.

## Additional Scenarios
- **Non-Interactive Prompts**: Use the `run` subcommand for quick queries.
- **Debugging**: Enable detailed logs with `-d` for troubleshooting AI interactions.

## Patterns & Examples

### Interactive Session
```bash
# Start an interactive AI session for the current project
x crush
```

### Non-Interactive Query
```bash
# Ask for a quick code explanation
x crush run "Explain the use of context in Go"
```

### Sandboxed/CWD Execution
```bash
# Run crush in a specific project directory with debug logging
x crush -d -c /path/to/project
```

## Checklist
- [ ] Ensure relevant AI API keys (OpenAI/Gemini) are set up via their respective x-cmd modules.
- [ ] Verify the target directory if using the `--cwd` flag.
- [ ] Confirm if YOLO mode is appropriate for the current task safety.
