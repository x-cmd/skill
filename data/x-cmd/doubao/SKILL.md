---
name: doubao
description: >
  Integrate Doubao (ByteDance Ark) AI models for chat and text processing.
  Core Scenario: When the user wants to use Doubao models for chat, reasoning, or text generation via CLI.
license: MIT
---

# doubao - Doubao AI Integration

The `doubao` module provides a CLI interface for ByteDance's Doubao models through the Volcengine Ark platform.

## When to Activate
- When the user wants to chat with Doubao models.
- When using specialized models like `doubao-seed`.
- When managing Doubao API keys and configurations.

## Core Principles & Rules
- **API Key Management**: Use `init` or `--cfg apikey=<key>` for setup.
- **Model Selection**: Use the `--model` flag to specify a specific Doubao model version.
- **Quick Access**: Use the `@doubao` alias for rapid interaction.

## Patterns & Examples

### Chat with Doubao
```bash
# Ask a question to the default Doubao model
@doubao "What are the benefits of using x-cmd?"
```

### Use Specific Model
```bash
# Use a specific seed model version
@doubao --model doubao-seed-1-8-251228 "Explain this code: ..."
```

## Checklist
- [ ] Ensure the Doubao API key is configured.
- [ ] Verify the chosen model name is correct.
