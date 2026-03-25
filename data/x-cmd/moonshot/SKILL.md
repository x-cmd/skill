---
name: moonshot
description: >
  Integrate Moonshot AI (Kimi) for chat and file-based context processing.
  Core Scenario: When the user wants to use Moonshot's large context capabilities for chat or file translation.
license: MIT
---

# moonshot - Moonshot AI (Kimi) Integration

The `moonshot` module provides access to Moonshot AI's services, known for their large context window and strong performance in Chinese language tasks.

## When to Activate
- When the user wants to chat with the Kimi model.
- When the user needs to analyze or translate files using Moonshot's context.
- When checking Moonshot account balance or managing file uploads.

## Core Principles & Rules
- **API Key Management**: Use `init` or `--cfg apikey=<key>` for setup.
- **Alias Support**: Use the `@kimi` alias for faster access.
- **Context Handling**: Leverage the large context window by attaching files with `--file`.

## Additional Scenarios
- **File Management**: Use the `file` subcommand to manage documents uploaded to Moonshot.
- **Balance Check**: Monitor credits using `x moonshot balance`.

## Patterns & Examples

### Translate Files with Kimi
```bash
# Use the @kimi alias to translate local files
@kimi --file ./content.en.md "Translate to Chinese"
```

### List Available Models
```bash
# View models supported by Moonshot
x moonshot model ls
```

## Checklist
- [ ] Ensure the Moonshot API key is configured.
- [ ] Verify that files attached exist and are readable.
