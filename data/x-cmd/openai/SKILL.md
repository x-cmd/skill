---
name: openai
description: >
  Integrate OpenAI API for AI capabilities including ChatGPT conversations, text generation, image creation, and audio conversion.
  Core Scenario: When the user needs to interact with OpenAI models for chat, translation, image generation, or audio processing.
license: MIT
---

# openai - OpenAI API Integration

The `openai` module provides a comprehensive CLI interface for OpenAI services, enabling users to perform complex AI tasks like text generation, translation, image creation, and audio conversion directly from the terminal.

## When to Activate
- When the user wants to use OpenAI for chat or text generation (e.g., generating commit messages).
- When the user needs to translate file contents using AI.
- When the user wants to generate images based on text prompts.
- When the user needs to convert text to speech or transcribe audio.
- When managing OpenAI API keys or model configurations.

## Core Principles & Rules
- **API Key Management**: Use `init` or `--cfg apikey=<key>` to set up the environment.
- **File Input**: Use the `--file` flag to provide context from local files to the chat/generation subcommands.
- **Reproducibility**: Be aware that AI outputs can vary; use appropriate parameters for consistent results if needed.

## Additional Scenarios
- **Git Commit Messages**: Pipe `git diff` into `@gpt` to generate standardized commit messages.
- **Audio Processing**: Use the `audio` subcommand for TTS (Text-to-Speech) or transcription tasks.
- **Embedding & Fine-tuning**: Advanced users can manage fine-tuned models or calculate text embeddings.

## Patterns & Examples

### Chat with File Context
```bash
# Translate multiple files to Chinese using OpenAI
x openai chat request --file ./abstract.en.md --file ./content.en.md "Translate to chinese"
```

### Image Generation
```bash
# Generate an image based on a prompt
x openai image create --prompt "a high-quality digital art of a futuristic city"
```

### Text to Speech
```bash
# Convert text to an audio file
x openai audio generate --input "Welcome to x-cmd" --model tts-1 --voice alloy
```

### Generate Commit Message
```bash
# Pipe diff into gpt for commit message generation
git diff | @gpt "generate a suitable Git commit message that follows the Conventional Commits format"
```

## Checklist
- [ ] Ensure the OpenAI API key is configured using `x openai init`.
- [ ] Verify the input files exist when using the `--file` flag.
- [ ] Confirm the desired model and voice for audio generation.
