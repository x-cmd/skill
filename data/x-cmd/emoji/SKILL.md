---
name: emoji
description: >
  Search, list, and manage emoji resources from the command line.
  Core Scenario: When the user needs to find specific emoji characters or browse emoji groups.
license: MIT
---

# emoji - Emoji Search & Listing

The `emoji` module provides an interface to browse and search for emoji characters. It categorizes emojis into groups and supports structured data exports.

## When to Activate
- When the user wants to search for specific emojis by keyword.
- When browsing emoji groups (e.g., smileys, nature).
- When exporting emoji lists to CSV or tables for use in documentation or apps.

## Core Principles & Rules
- **Categorization**: Emojis are grouped for easier navigation.
- **Resource Management**: Use `update` to ensure the local emoji database is current.

## Patterns & Examples

### Interactive Browse
```bash
# Open an interactive app to view all emojis
x emoji
```

### Search Group
```bash
# List all emojis in the 'smileys' group
x emoji ls smileys
```

### Export to Table
```bash
# List emojis in a structured table format
x emoji ls --table
```

## Checklist
- [ ] Confirm if the user is looking for a specific emoji or browsing a group.
- [ ] Verify if a specific output format (CSV/Table) is needed.
