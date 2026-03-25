---
name: deepseek
description: >
  Integrate DeepSeek AI for high-performance text generation and specialized reasoning.
  Core Scenario: When the user wants to use DeepSeek's V3 or Reasoner models for coding, translation, or complex logic.
license: MIT
---

# deepseek - DeepSeek AI Integration

The `deepseek` module provides a CLI interface for DeepSeek's AI models, including the powerful V3 model and the reasoning-focused model.

## When to Activate
- When the user wants to use DeepSeek for chat, text generation, or coding tasks.
- When the user needs complex reasoning or math solving using the `deepseek-reasoner` model.
- When checking DeepSeek account balance or managing API keys.

## Core Principles & Rules
- **API Key Management**: Use `init` or `--cfg apikey=<key>` for setup.
- **Model Selection**: Default is V3 (`@ds`). Use `--model deepseek-reasoner` for tasks requiring deep thought.
- **Balance Monitoring**: Use the `balance` subcommand to keep track of credits.

## Additional Scenarios
- **Quick Alias**: Use `@ds` for rapid interaction.
- **Project Configuration**: Set default models per project using the `--cfg` options.

## Patterns & Examples

### Chat with DeepSeek V3
```bash
# Ask a general question
@ds "Explain the concept of quantum entanglement"
```

### Logical Reasoning
```bash
# Use the reasoning model for complex logic
@ds --model deepseek-reasoner "Solve this complex probability problem: ..."
```

### Check Account Balance
```bash
# View current credits
x deepseek balance
```

## Checklist
- [ ] Ensure the DeepSeek API key is initialized.
- [ ] Confirm if the reasoning model is better suited for the current task.
