---
name: gtb
description: >
  Search and browse books from Project Gutenberg directly in the terminal.
  Core Scenario: When the user needs to find classic literature, read full book text, or search Gutenberg archives.
license: MIT
---

# gtb - Project Gutenberg CLI Browser

The `gtb` module provides an interface for Project Gutenberg, allowing users to search through its archive of free eBooks, view text content, and access metadata from the CLI.

## When to Activate
- When the user wants to search for classic books or authors (e.g., Shakespeare, Dumas).
- When reading the full text of a public domain book in the terminal.
- When accessing book metadata or viewing books in a web browser via CLI.

## Core Principles & Rules
- **Text-First**: Optimized for retrieving and viewing raw text content (`txt` subcommand).
- **Interactive TUI**: Use `show` for an interactive reading experience.
- **Search Support**: Leverages DuckDuckGo for broader queries related to Gutenberg content.

## Patterns & Examples

### Search by Author
```bash
# Search for books by Alexander Dumas
x gtb search Dumas
```

### View Book Text
```bash
# Retrieve the full text of book ID 100
x gtb txt 100
```

### Interactive Reading
```bash
# Browse book ID 100 in an interactive interface
x gtb show 100
```

## Checklist
- [ ] Confirm if the user has a specific book ID or needs to perform a search.
- [ ] Verify if the user wants to read in terminal or open in a browser.
