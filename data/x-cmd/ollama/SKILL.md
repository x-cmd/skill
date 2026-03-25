---
name: ollama
description: >
  CLI client for Ollama, the open-source framework for local LLM deployment.
  Core Scenario: When the user wants to pull, manage, or run local AI models (Llama 3, Mistral, etc.) via Ollama.
license: MIT
---

# ollama - Local LLM Management & Interaction

The `ollama` module provides an enhanced CLI interface for Ollama, enabling users to easily manage local models, perform translations, and interact with various AI architectures.

## When to Activate
- When the user wants to install or manage Ollama services.
- When pulling models from the Ollama registry (e.g., Llama 3, Mistral).
- When chatting with local models or using them for file-based translation.
- When using an interactive UI to browse and download models.

## Core Principles & Rules
- **Registry Interaction**: Use `pull` and `push` to sync models with the Ollama registry.
- **Convenience Aliases**: Support for `@o` alias for rapid local LLM chat.
- **File Context**: Leverage the `--file` flag to provide local context to models.

## Additional Scenarios
- **Interactive Browsing**: Use `la` for an interactive UI to explore available models.
- **Service Control**: Use `serve` to manually start the Ollama backend.

## Patterns & Examples

### Pull and Run
```bash
# Download and start a chat session with Mistral
x ollama pull mistral
@o "How does a vector database work?"
```

### Translate with File Context
```bash
# Translate local documents using a local model
@o --file ./abstract.en.md "Translate to Chinese"
```

### Interactive UI
```bash
# Browse the Ollama registry interactively
x ollama la
```

## Checklist
- [ ] Ensure the Ollama service is installed and running.
- [ ] Verify if the desired model has been pulled to local storage.
- [ ] Check compatibility of the `--file` input with the model's context window.
