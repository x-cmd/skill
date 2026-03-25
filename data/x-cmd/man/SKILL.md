---
name: man
description: >
  Enhanced man page viewer with FZF integration and AI-powered command explanation.
  Core Scenario: When the user needs to browse manual pages or requires an AI explanation of complex command options.
license: MIT
---

# man - Enhanced Manual Page Viewer

The `man` module extends the standard system manual capabilities by adding interactive browsing via FZF and integrating with AI services like ManKier to explain complex command strings.

## When to Activate
- When the user wants to search for man pages using keywords.
- When an interactive list of all system man pages is required.
- When the user needs an AI to explain the meaning of specific command flags (e.g., `tar -czvf`).
- When a quick community reference (TLDR style) is needed for a command via `x man :keyword`.

## Core Principles & Rules
- **Interactive Search**: Use `--fzf` for a searchable TUI experience.
- **AI Explanation**: Leverage the `--explain` flag to breakdown complex commands into readable summaries.
- **Hybrid Support**: Support for accessing TLDR content directly from within the `man` command using the `:` prefix.

## Patterns & Examples

### Fuzzy Search Manuals
```bash
# Interactively choose from all system manual pages
x man --fzf
```

### AI Command Explanation
```bash
# Use AI to explain exactly what these tar flags do
x man --explain "tar -czvf"
```

### Quick TLDR Style
```bash
# View the simplified TLDR reference for ssh
x man :ssh
```

## Checklist
- [ ] Confirm if the user needs the full manual or just a quick explanation.
- [ ] Verify if the command string for explanation is complete.
