---
name: llmf
description: >
  Run local LLMs with zero dependencies using llamafile, supporting API servers, CLI chat, and model management.
  Core Scenario: When the user wants to run high-performance local models (GGUF/llamafile) without external dependencies or cloud APIs.
license: MIT
---

# llmf - Zero-Dependency Local LLM Runner

The `llmf` module leverages llamafile technology to run large language models locally with zero setup. It provides a full suite of tools for serving, chatting, and managing models.

## When to Activate
- When the user wants to run an AI model locally without cloud API access.
- When setting up a local OpenAI-compatible API server (`serve`).
- When performing fast, one-off text generation tasks via CLI.
- When managing local GGUF or llamafile models (download, import, tokenize).

## Core Principles & Rules
- **Zero-Dependency**: Emphasize that models run locally without external runtimes.
- **Compatibility**: The `serve` command provides an OpenAI-compatible HTTP interface.
- **Resource Management**: Models are stored in `~/.x-cmd/data/llmf/model/`.

## Additional Scenarios
- **Token Analysis**: Use `tokenize` to breakdown text into token details.
- **Headless Server**: Start the API server without opening a browser using `--nobrowser`.

## Patterns & Examples

### Start local API Server
```bash
# Run a specific model as an OpenAI-compatible server
x llmf serve -m llava/v1.5-7b/q4_k.gguf --nobrowser
```

### Fast Text Generation
```bash
# Execute a single prompt and output the result
x llmf cli -p 'Write a concise summary of the Python GIL'
```

### Download Model
```bash
# Pull a specific model from the library
x llmf model download llava/v1.5-7b/q4_k.gguf
```

## Checklist
- [ ] Ensure the user has enough disk space for local models.
- [ ] Verify if the chosen model format (GGUF/llamafile) is supported.
- [ ] Check if the local API port is available if running `serve`.
