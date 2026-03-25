---
name: buse
description: >
  Enhancement module for browser-use, allowing AI to control browsers using natural language.
  Core Scenario: When the user wants an AI agent to perform tasks in a browser (e.g., searching docs, taking screenshots) via terminal.
license: MIT
---

# buse - Natural Language Browser Automation

The `buse` module enables users to control a web browser using natural language through the `browser-use` framework. It automates complex browser actions without requiring manual coding.

## When to Activate
- When the user wants an AI to navigate websites or interact with web elements.
- When performing automated browser tasks like documentation search or visual verification (screenshots).
- When using a terminal UI to interactively guide a browser agent.
- When running headless browser tasks for data extraction or testing.

## Core Principles & Rules
- **Safety & Permissions**: AI-generated browser actions should be monitored; use caution with sensitive sites.
- **Model Requirement**: Requires an AI model API key (OpenAI, Gemini, etc.) configured via respective x-cmd modules.
- **Context Handling**: Supports custom browser windows, user data directories, and CDP connections.

## Additional Scenarios
- **Visual Evidence**: Automatically take screenshots during a browser task using specific prompts.
- **MCP Mode**: Run as a Model Context Protocol server for integration with other AI tools.

## Patterns & Examples

### Interactive Browser Control
```bash
# Launch the interactive terminal UI for browser control
x buse
```

### Direct Task Execution
```bash
# Run a specific browser task non-interactively
x buse -p "Search for OpenAI documentation and take a screenshot of the homepage"
```

### Headless Mode
```bash
# Run a task in the background without a visible window
x buse --headless -p "Check the stock status of an item on ExampleStore.com"
```

## Checklist
- [ ] Ensure browser-use and Chromium are installed via `x buse --install`.
- [ ] Confirm that at least one AI model API key is set up.
- [ ] Verify if the task requires a visible window or can run headless.
