---
name: jina
description: >
  CLI tools for Jina.ai, providing webpage reading, embedding generation, and reranking.
  Core Scenario: When the user needs to fetch webpage content as Markdown, generate text embeddings, or perform information retrieval (reranking).
license: MIT
---

# jina - AI Search & Information Retrieval

The `jina` module integrates Jina.ai services into the terminal, offering powerful tools for RAG (Retrieval-Augmented Generation) workflows, including webpage reading and vector operations.

## When to Activate
- When the user wants to fetch the content of a URL as Markdown or HTML.
- When generating vector embeddings for strings or local files.
- When performing semantic search or reranking results based on similarity.
- When managing Jina API keys and exploring available models.

## Core Principles & Rules
- **Markdown Conversion**: Use the `reader` subcommand (or direct URL) to get clean Markdown for AI consumption.
- **Vector Operations**: Use `embed` for generating embeddings and `reranker` for information retrieval.
- **Piping Support**: Highly compatible with terminal pipes for streaming data.

## Additional Scenarios
- **Similarity Search**: Compare files or strings using the `reranker` subcommand to find the best match.
- **Batch Processing**: Generate embeddings for entire files or multiple inputs simultaneously.

## Patterns & Examples

### Read Webpage as Markdown
```bash
# Fetch and display clean Markdown from a website
x jina https://x-cmd.com
```

### Generate Embeddings
```bash
# Create vector data for a text string
x jina embed generate "How does RAG work?"
```

### Rerank Results
```bash
# Find the top 3 most similar sentences in a file
x jina reranker generate -f ./LICENSE --sep "\n" --top 3 "how are you"
```

## Checklist
- [ ] Ensure the Jina API key is configured.
- [ ] Verify the target URL is accessible for the reader tool.
- [ ] Check if the chosen embedding or reranking model is active.
