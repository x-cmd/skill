---
name: wkp
description: >
  Wikipedia search and summary extraction tool.
  Core Scenario: When AI needs to quickly obtain definitions or official summaries in the terminal.
license: MIT
---

# x wkp - Wikipedia Assistant (AI Optimized)

`x wkp` provides a minimal interface for retrieving article lists, suggestions, and detailed summaries from Wikipedia via the command line.

## When to Activate
- When a quick definition of a term, historical event, or technical concept is needed.
- When retrieving the plain-text summary of a specific Wikipedia entry.
- When getting search suggestions or related article lists via keywords.

## Core Principles & Rules
- **Non-interactive First**: Avoid the `--app` interactive UI; use `extract` or `hop` subcommands directly for plain text.
- **Structured Retrieval**: Prioritize `extract` for detailed body text, or `hop` for a concise summary of the first matching item.

## Patterns & Examples

### Extract Detailed Summary
```bash
# Get a detailed summary for "OpenAI"
x wkp extract OpenAI
```

### Search for Related Entries
```bash
# Search for the keyword "Large Language Model"
x wkp search "Large Language Model"
```

### Hop to the First Result's Summary
```bash
# Search and output the summary of the first match (most efficient)
x wkp hop "Rust Programming"
```

### Get Search Suggestions
```bash
# Get suggestions for related entries when unsure of exact spelling
x wkp suggest "Quantom Computing"
```

## Checklist
- [ ] Confirm if the query term needs to be quoted.
- [ ] Choose between `search` (list) and `extract` (content) based on needs.
- [ ] Default to English or Chinese queries for the most comprehensive info.
