---
name: cht
description: >
  CLI client for cheat.sh, providing concise "cheat sheets" and code snippets for developers.
  Core Scenario: When the user needs short, practical examples for programming languages or CLI tools.
license: MIT
---

# cht - Developer Cheat Sheets & Snippets

The `cht` module leverages the cheat.sh platform to provide high-quality, community-vetted code snippets and command usage examples. It is designed for developers who need "how-to" answers instantly.

## When to Activate
- When the user needs a code example for a specific programming language (e.g., `python/hello`).
- When seeking a concise "cheat sheet" for a command (e.g., `ls`).
- When the user wants to learn a language quickly using "learn-x-in-minutes" content (`--learn`).
- When searching across multiple cheat sheets using keywords.

## Core Principles & Rules
- **Example-Driven**: Focus on providing runnable code snippets or direct command examples.
- **Breadth of Content**: Covers hundreds of CLI tools and dozens of programming languages.
- **Search Capability**: Use `-s` for keyword-based search within the cheat.sh ecosystem.

## Patterns & Examples

### Language Snippet
```bash
# Get a Python 'hello world' example from cheat.sh
x cht python/hello
```

### Command Cheat Sheet
```bash
# View a concise summary of common 'sed' patterns
x cht sed
```

### Quick Learning
```bash
# Display the beginner cheat sheet for the Go language
x cht --learn go
```

## Checklist
- [ ] Confirm if the user needs a command reference or a language snippet.
- [ ] Verify if a specific task (like `hello`) should be appended to the language query.
