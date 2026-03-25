---
name: gg
description: >
  Search the web using Google Gemini AI, combining search results with AI-generated answers.
  Core Scenario: When the user needs real-time information from the web with AI-summarized responses and source citations.
license: MIT
---

# gg - AI Web Search & Synthesis

The `gg` module leverages Google Gemini AI to perform real-time web searches, providing synthesized answers that combine the power of Google Search with AI's understanding.

## When to Activate
- When the user asks questions that require up-to-date web information (e.g., current events, pricing, latest tech).
- When a summary of multiple web sources is needed.
- When the user wants to see the sources (URLs) for an AI-generated answer.
- When comparing information across different AI models (Gemini Pro vs. Flash).

## Core Principles & Rules
- **Formatting**: Defaults to Markdown (`--md`) in TTY for readability.
- **Source Citation**: Use `--source` or `--source-detail` if the user needs to verify the origins of the info.
- **Cache Management**: Results are cached for 1 hour by default to save tokens; use `--cache-time` to override.
- **Model Choice**: Use `--model` to switch between high-reasoning (Pro) and high-speed (Flash) versions.

## Patterns & Examples

### Basic Question
```bash
# Ask a general question about current facts
x gg "Who won the Euro 2024?"
```

### Source-Detailed Search
```bash
# Search for documentation with full source details
x gg --source-detail "Python tutorial for beginners"
```

### Specific Model
```bash
# Use the high-reasoning model for complex math or logic
x gg --model gemini-2.5-pro "Latest breakthroughs in quantum computing"
```

## Checklist
- [ ] Confirm if real-time web info is required for the query.
- [ ] Verify if the user wants source links included in the response.
- [ ] Check if a specific model (Pro/Flash) is better suited for the task.
