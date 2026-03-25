---
name: wttr
description: >
  Check weather forecasts and moon phases directly from the terminal.
  Core Scenario: When the user needs current weather information or upcoming forecasts via terminal.
license: MIT
---

# wttr - CLI Weather Forecast

The `wttr` module provides weather information for current or specified locations, utilizing the wttr.in service. It supports human-readable formats, moon phases, and unit customization.

## When to Activate
- When the user wants to check the weather at their current location or a specific city.
- When viewing current or historical moon phases.
- When needing a concise weather summary for scripts or status lines.

## Core Principles & Rules
- **Localization**: Automatically adapts units and language based on the environment unless specified.
- **Moon Phases**: Use the `moon` subcommand for lunar data.

## Patterns & Examples

### Check Local Weather
```bash
# View the weather forecast for the current location
x wttr
```

### Check Specific City
```bash
# View the weather for Beijing
x wttr Beijing
```

### Moon Phase
```bash
# View the current phase of the moon
x wttr moon
```

## Checklist
- [ ] Confirm the target location (default is current).
- [ ] Verify if specific units (metric/imperial) are required.
