---
name: sed
description: >
  Enhanced stream editor with interactive preview of search and replace operations using FZF.
  Core Scenario: When the user needs to perform text substitutions and wants to see the effects before applying changes.
license: MIT
---

# sed - Stream Editor with Interactive Preview

The `sed` module extends the classic stream editor by adding an interactive preview mode. Users can test their regex substitutions and see context differences (`ctrl-s`) in real-time before modifying files.

## When to Activate
- When the user wants to perform text search and replace operations.
- When an interactive preview of regex effects is needed to avoid errors.
- When performing bulk line deletions or insertions in files.
- When using extended regular expressions for complex text manipulation.

## Core Principles & Rules
- **Preview First**: Encourage using the `--fzfapp` mode to verify changes visually.
- **Context Differences**: Use `ctrl-s` within the interactive app to view diffs.
- **In-place Safety**: Use `-i` carefully, preferably with a backup suffix.

## Patterns & Examples

### Interactive Preview
```bash
# Interactively test and preview regex changes on a file
x sed --fzfapp test.txt
```

### Basic Substitution
```bash
# Replace 'world' with 'universe' in a piped string
echo "hello world" | x sed 's/world/universe/'
```

### Global Replace
```bash
# Replace all occurrences of 'foo' with 'bar' in a file
x sed 's/foo/bar/g' file.txt
```

## Checklist
- [ ] Confirm if the user needs an interactive preview or direct execution.
- [ ] Verify the regex pattern for search and replace.
- [ ] Check if the file should be modified in-place or output to stdout.
