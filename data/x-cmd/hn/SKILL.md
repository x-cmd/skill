---
name: hn
description: >
  Browse Hacker News from the CLI with interactive TUI, AI search integration, and user analysis.
  Core Scenario: When the user wants to read top HN posts, search HN via AI, or analyze HN user stats.
license: MIT
---

# hn - Hacker News CLI Browser

The `hn` module provides an interactive terminal interface for browsing Hacker News. It supports viewing top/new/best stories, AI-powered searching, and analyzing user metrics like h-index.

## When to Activate
- When the user wants to browse Hacker News stories (top, new, ask, etc.) in an interactive table.
- When searching for specific topics on Hacker News using DuckDuckGo or AI (`::` prefix).
- When retrieving details for a specific post ID or user.
- When exporting HN data to JSON for scripting or analysis.

## Core Principles & Rules
- **Interactive Browsing**: Optimized for terminal navigation with shortcuts to open links.
- **AI-Enhanced Search**: Use `ddgoai` or `::` to combine search results with AI summaries.
- **User Metrics**: Provides tools like `hidx` to calculate user impact.

## Patterns & Examples

### Browse Top Stories
```bash
# Open interactive TUI for top HN stories
x hn
```

### AI Search HN
```bash
# Search for 'llama3' on HN with AI-generated summaries
x hn :: llama3
```

### View User Info
```bash
# Get details and h-index for a specific HN user
x hn user dang
x hn hi dang
```

## Checklist
- [ ] Confirm if the user wants to browse categories or perform a search.
- [ ] Verify if AI-enhanced search is preferred for complex queries.
