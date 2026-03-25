---
name: gemini
description: >
  Integrate Google Gemini AI for chat, text analysis, and multimodal tasks like image understanding.
  Core Scenario: When the user wants to use Gemini models for advanced chat, translation with file context, or analyzing images via CLI.
license: MIT
---

# gemini - Google Gemini AI Integration

The `gemini` module provides a powerful CLI interface for Google's Gemini AI models, supporting text generation, file-based context, and multimodal inputs like images.

## When to Activate
- When the user wants to use Gemini for chat or complex text generation.
- When the user needs to analyze or translate files using Gemini's large context window.
- When the user wants to provide images for AI analysis via the terminal.
- When managing Gemini API keys or exploring available models.
- When using Gemini's integrated Google Search tool (`gg`).

## Core Principles & Rules
- **API Key Management**: Use `init` or `--cfg apikey=<key>` for setup.
- **Multimodal Support**: Use the `--file` flag to attach images or documents for the AI to process.
- **Token Counting**: Use the `--count` flag to estimate token usage before or during a request.
- **Google Search Integration**: Utilize the `gg` subcommand to leverage real-time information from Google Search within the AI response.

## Additional Scenarios
- **Git Commit Messages**: Pipe `git diff` into `@gemini` for high-quality, standardized commit messages.
- **Combined Tools**: Pipe outputs from other tools (e.g., `x wkp` for Wikipedia) into `@gemini` for specialized analysis.
- **Official CLI**: Access the official Google Gemini CLI via `x gemini cli`.

## Patterns & Examples

### Chat with Image and Text
```bash
# Ask Gemini to describe an image file
@gemini --file ./pic.jpg "What is described in this image?"
```

### Translate Files
```bash
# Translate multiple local files using Gemini's context
x gemini chat request --file ./abstract.en.md --file ./content.en.md "Translate these to Chinese"
```

### Google Search Integration
```bash
# Ask a question and use Google Search for the answer
x gemini gg "What are the latest features in x-cmd?"
```

## Checklist
- [ ] Ensure the Gemini API key is configured using `x gemini init`.
- [ ] Verify that attached files (images/documents) exist and are supported formats.
- [ ] Confirm the specific model version using `x gemini model ls` if needed.
