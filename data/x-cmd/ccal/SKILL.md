---
name: ccal
description: >
  Terminal calendar tool for the Chinese lunar calendar, solar terms, and traditional festivals.
  Core Scenario: When the user needs to check lunar dates, solar terms, or traditional Chinese holiday information.
license: MIT
---

# ccal - Chinese Lunar Calendar

The `ccal` module is a terminal calendar specifically designed for users interested in the Chinese lunar calendar. It provides lunar dates, solar terms, and information about traditional festivals and daily fortune (Huangli).

## When to Activate
- When the user wants to see the current or a specific date's lunar info.
- When checking solar terms, public holidays, or makeup work days in China.
- When viewing traditional festival dates or zodiac information interactively.

## Core Principles & Rules
- **Interactive Browsing**: Use `fz` subcommands for a searchable TUI experience.
- **Detailed Info**: Use the `info` subcommand for a comprehensive breakdown of a single day (lunar, stems-and-branches, etc.).
- **Localization**: Supports customizing weekday naming (e.g., using "礼拜" vs "星期").

## Patterns & Examples

### Today's Lunar Info
```bash
# View detailed lunar and traditional info for today
x ccal info
```

### Specific Date
```bash
# Check lunar info for a future date
x ccal info 2025-5-20
```

### Interactive Year View
```bash
# Browse traditional festivals and zodiacs interactively for the year
x ccal fzy
```

## Checklist
- [ ] Confirm if the user needs info for today or a specific date.
- [ ] Verify if an interactive view or a simple data output is preferred.
