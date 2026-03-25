---
name: humantime
description: >
  Convert between seconds and human-readable time formats (e.g., 1h30m).
  Core Scenario: When the user needs to parse or display durations in scripts or terminal output.
license: MIT
---

# humantime - Human-Friendly Time Conversion

The `humantime` module simplifies time manipulation by converting raw seconds into readable formats (like `1h 30m`) and vice-versa. It supports fractional seconds and automatic type detection.

## When to Activate
- When displaying script execution duration to users.
- When calculating cache expiration times based on human-readable input.
- When checking if a duration falls within a specific limit (`in` subcommand).

## Core Principles & Rules
- **Auto-Detection**: Inputting an integer (e.g., `3600`) converts to time format (`1h`), while a time string (e.g., `5m`) converts to seconds (`300`).
- **Time Units**: Supports `d` (days), `h` (hours), `m` (minutes), and `s` (seconds).
- **Fractional Precision**: Handles decimal seconds (e.g., `3661.5s = 1h1m1.5s`) accurately.

## Patterns & Examples

### Convert Seconds to Readable
```bash
# Convert one day into readable format
x humantime 86400
```

### Convert Readable to Seconds
```bash
# Get raw seconds for 2 hours and 30 minutes
x humantime 2h30m
```

### Within Range Check
```bash
# Check if 240 seconds is within 5 minutes (returns 0 for yes)
x humantime in 5m 240
```

## Checklist
- [ ] Confirm if the input is an integer (seconds) or a string (readable).
- [ ] Verify if high precision (fractional seconds) is required.
