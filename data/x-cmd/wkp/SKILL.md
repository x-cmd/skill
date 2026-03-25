---
name: wkp
description: >
  Search Wikipedia and retrieve article extracts or summaries directly from the terminal.
  Core Scenario: When the user needs to quickly lookup terms or get summaries of Wikipedia entries.
license: MIT
---

# wkp - Wikipedia Search & Extract Utility

The `wkp` module allows users to search Wikipedia and extract text content from articles. It supports interactive browsing, automatic suggestions, and DuckDuckGo integration.

## When to Activate
- When the user wants a quick summary or extract of a specific Wikipedia topic.
- When searching for articles by title or using fuzzy suggestions for typos.
- When wanting to open a Wikipedia page directly in the browser from the CLI.

## Core Principles & Rules
- **Conciseness**: Prioritize `extract` or `hop` for immediate text summaries in the terminal.
- **Search Assistance**: Use `suggest` to help users find the correct term for their query.
- **Interactive UI**: Use `--app` mode for exploring search results.

## Patterns & Examples

### Fetch Summary
```bash
# Get a concise summary of 'OpenAI'
x wkp extract OpenAI
```

### Direct Search and Summary
```bash
# Search for 'Python' and display the first result's extract
x wkp hop Python
```

### Suggestions
```bash
# Get Wikipedia suggestions for a misspelled term
x wkp suggest pythen
```

## Checklist
- [ ] Confirm if the user needs a text extract or a full search result list.
- [ ] Verify if the term needs automatic suggestions for better accuracy.
