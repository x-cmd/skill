---
name: gcal
description: >
  Enhanced interface for GNU gcal, supporting holiday configurations and custom calendar formats.
  Core Scenario: When the user needs a highly configurable calendar with national holiday support.
license: MIT
---

# gcal - GNU gcal Enhancement

The `gcal` module provides a powerful wrapper around GNU gcal, making it easy to display calendars with localized holiday support and customizable formats.

## When to Activate
- When the user wants to print monthly or yearly calendars.
- When needing holiday information for specific countries (e.g., China).
- When initializing or managing calendar preference settings.

## Core Principles & Rules
- **Holiday Awareness**: Use the `cc` subcommand to list and select national holiday codes.
- **Persistent Config**: Use `cfg` to maintain settings like country codes across sessions.

## Patterns & Examples

### Print Current Month
```bash
# Display the calendar for the current month
x gcal
```

### Yearly Calendar
```bash
# Display the full calendar for 2025
x gcal 2025
```

### Configure Holidays
```bash
# Set gcal to use Chinese public holidays
x gcal --cfg cc=CN
```

## Checklist
- [ ] Confirm if a monthly or yearly view is needed.
- [ ] Verify if a specific country's holidays should be included.
