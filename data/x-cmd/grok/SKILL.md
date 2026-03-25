---
name: grok
description: >
  Integrate xAI Grok for intelligent chat and real-time information processing.
  Core Scenario: When the user wants to use Grok models for advanced reasoning or real-time query answers via CLI.
license: MIT
---

# grok - xAI Grok Integration

The `grok` module provides a CLI interface for xAI's Grok models, supporting powerful text generation and reasoning.

## When to Activate
- When the user wants to chat with Grok.
- When performing real-time queries or complex reasoning tasks.
- When managing xAI API keys and model configurations.

## Core Principles & Rules
- **API Key Management**: Use `init` or `--cfg apikey=<key>` for setup.
- **Experimental Status**: Be aware that this module is currently in an experimental phase.
- **Alias Access**: Use the `@grok` alias for rapid interaction.

## Patterns & Examples

### Chat with Grok
```bash
# Ask Grok a question via terminal
@grok "What are the latest breakthroughs in space exploration?"
```

## Checklist
- [ ] Ensure the xAI API key is configured.
- [ ] Confirm if the current task requires Grok's specific reasoning style.
