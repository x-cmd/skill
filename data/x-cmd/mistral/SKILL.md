---
name: mistral
description: >
  Integrate Mistral AI for efficient language modeling and text generation tasks.
  Core Scenario: When the user wants to use Mistral models for chat or translation via CLI.
license: MIT
---

# mistral - Mistral AI Integration

The `mistral` module provides a CLI interface for Mistral AI's services, supporting high-efficiency text generation and multi-language translation.

## When to Activate
- When the user wants to chat with Mistral models.
- When the user needs to translate file content using Mistral's API.
- When checking or managing Mistral API keys.

## Core Principles & Rules
- **API Key Management**: Use `init` or `--cfg apikey=<key>` for setup.
- **Alias Access**: Use the `@mistral` alias for fast interaction.
- **Model Exploration**: Use `model ls` to see all available Mistral models.

## Patterns & Examples

### Translate Files
```bash
# Use Mistral to translate a file to Chinese
@mistral --file ./document.en.md "Translate to Chinese"
```

### List Models
```bash
# View all supported Mistral models
x mistral model ls
```

## Checklist
- [ ] Ensure the Mistral API key is configured.
- [ ] Verify file paths when using the `--file` flag.
