---
name: jina
description: >
  Jina AI enhancement tool: Web parsing, search, and reranking.
  Core Scenario: When AI needs to read a URL's content, perform real-time web searches, or rank text segments by semantic relevance.
license: MIT
---

# x jina - AI Knowledge Assistant (AI Optimized)

`x jina` serves as the "eyes" and "external brain" for AI Agents. Its core strength lies in converting complex web pages into AI-friendly Markdown and providing powerful search and semantic ranking capabilities.

## When to Activate
- When reading and analyzing the content of a specific web page (URL), automatically converted to Markdown.
- When performing real-time web searches to gather the latest information.
- When identifying the most semantically relevant segments from a set of documents (Reranker).

## Core Principles & Rules
- **Markdown First**: Retrieves web pages in Markdown by default, which is optimal for LLM consumption.
- **Environment Requirements**: Requires a Jina API Key. If not configured, AI should guide the user through initialization.
- **Configuration Guidance**:
  - Direct the user to jina.ai to obtain a free or paid API Key.
  - Suggest the user run `x jina init` for configuration.

## Patterns & Examples

### Read Web Content (Most Common)
```bash
# Retrieve web content and convert to Markdown for AI reading
x jina reader https://example.com/article
```

### Real-time Web Search
```bash
# Search for keywords and return summaries of the top 5 results
x jina search "latest AI trends 2024"
```

### Semantic Reranking
```bash
# Find the top 3 segments from a file most relevant to "how to install"
x jina reranker generate -f docs.txt --top 3 "how to install"
```

### Text Embedding
```bash
# Generate vector data for text, useful for vector database retrieval
x jina embed generate "This is a test text"
```

## Configuration Guide (for AI)
If an API error occurs, provide this guidance to the user:
> Please obtain an API Key from Jina AI (https://jina.ai), then run the following command in your terminal to initialize:
> `x jina init`

## Checklist
- [ ] Prioritize using the `reader` subcommand for Markdown.
- [ ] Combine search tasks with `reranker` for better precision.
