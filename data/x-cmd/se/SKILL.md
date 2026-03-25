---
name: se
description: >
  Search and browse Stack Exchange (Stack Overflow, Ask Ubuntu, etc.) from the command line.
  Core Scenario: When the user needs to find answers to technical questions or browse Stack Overflow solutions.
license: MIT
---

# se - Stack Exchange CLI Browser

The `se` module allows users to search the entire Stack Exchange network, including Stack Overflow and Ask Ubuntu, and browse questions and answers directly in the terminal.

## When to Activate
- When the user has a specific technical problem and wants to search Stack Overflow.
- When browsing answers for a specific question ID.
- When searching platform-specific sites like Ask Ubuntu for system-related issues.

## Core Principles & Rules
- **Site Selection**: Support for site-specific prefixes like `:so` (Stack Overflow) and `:au` (Ask Ubuntu).
- **Interactive TUI**: Use `--app` to browse question answers in a structured terminal UI.
- **Search Integration**: Uses DuckDuckGo for broader search within the SE ecosystem if needed.

## Patterns & Examples

### Search Stack Overflow
```bash
# Search for JSON parsing issues in Python on Stack Overflow
x se :so "python json parse error"
```

### View Answers
```bash
# Get all answers for a specific question ID
x se question --showall 75261408
```

### Interactive App
```bash
# Browse question answers in an interactive TUI
x se question --app 75261408
```

## Checklist
- [ ] Confirm if the search should be limited to a specific SE site (e.g., Stack Overflow).
- [ ] Verify if the user has a specific question ID or needs to perform a keyword search.
- [ ] Check if the interactive view or raw output is preferred.
