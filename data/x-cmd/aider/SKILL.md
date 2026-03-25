---
name: aider
description: >
  AI pair-programming tool that allows you to code with LLMs directly in your terminal and Git repository.
  Core Scenario: When the user wants to collaborate with an AI on code modifications, automated commits, or whole-repo context tasks.
license: MIT
---

# aider - AI Pair-Programming in Terminal

`aider` is an AI pair-programming tool that enables developers to collaborate with high-end LLMs (Claude 3.7, DeepSeek R1, GPT-4o, etc.) directly in the terminal. It understands entire Git repositories and can perform multi-file changes and automated commits.

## When to Activate
- When the user wants to start an AI-assisted pair-programming session.
- When performing complex code refactoring or modifications across multiple files.
- When the user needs automated, semantic Git commit messages for AI-generated changes.
- When utilizing "Voice-to-Code" or embedding external content (screenshots/webpages) into the coding context.

## Core Principles & Rules
- **Git Integration**: Aider is most effective when run inside a Git repository.
- **Model Versatility**: Supports a wide range of models (Claude, DeepSeek, OpenAI, local LLMs).
- **Automation**: Can automatically run lints and tests to verify changes.

## Additional Scenarios
- **Whole-Repo Context**: Automatically generates a map of the repository to help the LLM understand complex architectures.
- **Multi-File Editing**: Seamlessly handles modifications that span multiple source files.

## Patterns & Examples

### Launch Aider
```bash
# Start an interactive pair-programming session
x aider
```

## Checklist
- [ ] Verify that the user is inside a Git repository for optimal performance.
- [ ] Confirm the target LLM is configured and accessible.
- [ ] Check if additional dependencies (like for voice input) are needed.
