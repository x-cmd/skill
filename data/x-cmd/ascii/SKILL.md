---
name: ascii
description: >
  Handle ASCII-related tasks including tables, art fonts, image conversion, and terminal rendering (Mermaid).
  Core Scenario: When the user needs to view ASCII codes, generate ASCII art, or render diagrams like Mermaid in terminal.
license: MIT
---

# ascii - ASCII Utility & Terminal Arts

The `ascii` module provides a set of tools for working with ASCII text and visual assets. It covers everything from standard ASCII tables to complex terminal renderings like world maps and Mermaid diagrams.

## When to Activate
- When the user wants to look up ASCII character codes (decimal, hex).
- When generating ASCII art text (cfont) or converting images to ASCII art.
- When rendering Mermaid diagrams or drawing ASCII line graphs in the terminal.
- When viewing an ASCII-based world map or running ASCII animations (fireworks).

## Core Principles & Rules
- **Dynamic Execution**: Many subcommands (like `cfont` and `mermaid`) download the necessary tools on-demand via deno or npm.
- **Visual Creativity**: Supports styling art fonts with colors and alignments.
- **Piping Support**: Can draw graphs from piped numeric data (e.g., `seq 1 10 | x ascii graph`).

## Patterns & Examples

### ASCII Table
```bash
# View the standard ASCII table
x ascii table
```

### ASCII Art Font
```bash
# Convert text to a colorful ASCII art font
x ascii cfont x-cmd -g red,blue
```

### Mermaid Rendering
```bash
# Render a Mermaid chart directly in the terminal
x ascii mermaid
```

### Data Graphing
```bash
# Create an ASCII line graph from a sequence
seq 1 10 | x ascii graph
```

## Checklist
- [ ] Confirm if the user needs a code reference or a visual transformation.
- [ ] Verify if specific colors or styles are requested for ASCII art.
- [ ] Ensure terminal width is sufficient for large ASCII renderings.
