---
name: ddgo
description: >
  CLI tool for DuckDuckGo search, supporting result summaries and AI-driven answer extraction.
  Core Scenario: When the user needs to perform web searches or get AI-summarized answers from DuckDuckGo results.
license: MIT
---

# ddgo - DuckDuckGo CLI Search

The `ddgo` module provides a powerful terminal-based interface for DuckDuckGo, enabling users to perform web searches, extract results in structured formats, and leverage AI for summarizing answers.

## When to Activate
- When the user wants to perform a privacy-focused web search via CLI.
- When needing an AI-generated summary of search results for a specific query.
- When exporting search results to JSON for further processing.
- When performing site-specific searches (e.g., `site:x-cmd.com`).

## Core Principles & Rules
- **AI Integration**: Use the `--ai` flag to automatically select and summarize the most relevant search result.
- **Data-Friendly**: Supports structured JSON output for script integration.
- **Structured Viewing**: Use `dump --app` for an interactive table view of search results.

## Patterns & Examples

### AI-Summarized Search
```bash
# Get an AI summary of search results for 'bash tips'
x ddgo --ai "bash best practices"
```

### Site-Specific Search
```bash
# Search for 'jq' content on the x-cmd.com website
x ddgo dump --json "site:x-cmd.com jq"
```

### Top Results
```bash
# Retrieve the top 10 search results for a query
x ddgo --top 10 "python threading tutorial"
```

## Checklist
- [ ] Confirm if the user needs a general search or an AI-summarized answer.
- [ ] Verify if the results should be limited to a specific site or number.
- [ ] Check if JSON output is required for subsequent tasks.
