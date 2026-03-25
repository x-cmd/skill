---
name: sd
description: >
  Modern finding and replacing tool with intuitive regex syntax and interactive FZF preview.
  Core Scenario: When the user needs a faster, more readable alternative to sed for text substitutions.
license: MIT
---

# sd - Intuitive Find & Replace Utility

The `sd` module is a modern alternative to `sed`, focusing on substitution with a more readable syntax. It integrates with FZF for real-time previews of text changes and works without local installation via x-cmd pkg.

## When to Activate
- When the user wants a more intuitive syntax for text replacement.
- When needing interactive previews of search and replace effects.
- When performing fast string-based replacements without complex sed escaping.

## Core Principles & Rules
- **Readability**: Prioritize `sd` over `sed` for tasks that only require straightforward substitution.
- **Safety**: Supports `--preview` to output results to stdout without touching the file.
- **Interactive App**: Use `--fzfapp` for visual confirmation of changes.

## Patterns & Examples

### Interactive Search and Replace
```bash
# Open interactive FZF preview for text substitution
x sd --fzfapp test.txt
```

### String Replacement (Literal)
```bash
# Replace strings literally without treating them as regex
x sd -F "FIND_THIS" "REPLACE_WITH" file.txt
```

## Checklist
- [ ] Confirm if the find/replace terms should be treated as literal strings (`-F`).
- [ ] Verify if an interactive preview is required.
- [ ] Ensure the user is aware the file will be modified unless in preview mode.
