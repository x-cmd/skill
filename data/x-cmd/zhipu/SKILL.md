---
name: zhipu
description: >
  Integrate Zhipu AI (GLM) for advanced language modeling and image understanding.
  Core Scenario: When the user wants to use Zhipu's GLM models for chat, reasoning, or complex text tasks.
license: MIT
---

# zhipu - Zhipu AI (GLM) Integration

The `zhipu` module provides a CLI interface for Zhipu AI's GLM (General Language Model) series, supporting powerful text generation and reasoning.

## When to Activate
- When the user wants to use Zhipu GLM models for chat or text analysis.
- When managing Zhipu API keys and session defaults.
- When using the `@glm` alias for quick access.

## Core Principles & Rules
- **API Key Management**: Use `init` or `--cfg apikey=<key>` for setup.
- **Model Selection**: Specify models using the `--model` flag (e.g., `glm-4`).
- **Alias Access**: Prefer the `@glm` alias for a faster CLI experience.

## Patterns & Examples

### Chat with GLM
```bash
# Ask a question using the GLM model
@glm "What is the history of the Great Wall?"
```

### Use Specific Version
```bash
# Use a specific model version for reasoning
@glm --model glm-4.7 "How many Rs are there in the word strawberry?"
```

## Checklist
- [ ] Ensure the Zhipu API key is configured using `x zhipu init`.
- [ ] Confirm the desired GLM model version.
