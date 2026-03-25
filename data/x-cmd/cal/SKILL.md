---
name: cal
description: >
  Standard Gregorian calendar tool with support for zodiac signs and weekly formats.
  Core Scenario: When the user needs to view standard calendars or query zodiac signs by date.
license: MIT
---

# cal - Enhanced Gregorian Calendar

The `cal` module is an enhanced version of the classic calendar command, supporting standard Gregorian views along with extras like zodiac sign lookups.

## When to Activate
- When the user wants to see a standard monthly or yearly calendar.
- When querying the zodiac sign for a specific birth date.

## Core Principles & Rules
- **Localization**: Automatically outputs zodiac names in the preferred language (e.g., "狮子座" vs "Leo").
- **Simplicity**: Focuses on being a fast, drop-in replacement for `cal`.

## Patterns & Examples

### Query Zodiac
```bash
# Find the zodiac sign for a specific date
x cal zodiac 06-22
```

### View Future Calendar
```bash
# Show the calendar for December 2024
x cal 12 2024
```

## Checklist
- [ ] Confirm if the user needs a calendar view or a zodiac query.
