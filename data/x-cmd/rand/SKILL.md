---
name: rand
description: >
  Random data generator for identifiers, numbers, strings, and mock data.
  Core Scenario: When the user needs to generate UUIDs, random strings, or test data (emails, IPs) for scripting.
license: MIT
---

# rand - Random Data Generation

The `rand` module generates various types of random data, ranging from basic numbers and strings to complex identifiers like UUIDv7 and mock data like emails or IP addresses.

## When to Activate
- When generating unique identifiers (UUID, UUIDv7).
- When creating random passwords or tokens (strings, alphanum).
- When generating mock data for testing (emails, IPs).
- When requiring random integers or floats within a specific range.

## Core Principles & Rules
- **Range Control**: Use specific min/max values for `int` and `float` to suit the task.
- **Bulk Generation**: Support for generating multiple items simultaneously (e.g., `x rand email 5`).
- **Identifier Uniqueness**: Prefer `uuidv7` for time-ordered unique identifiers.

## Patterns & Examples

### Generate Identifiers
```bash
# Create a standard UUID and a UUIDv7
x rand uuid
x rand uuidv7
```

### Random Strings
```bash
# Generate a 16-character random alphanumeric string
x rand alphanum 16
```

### Mock Testing Data
```bash
# Generate 10 random IPv4 addresses
x rand ip 10
```

## Checklist
- [ ] Confirm the required length or range for the random data.
- [ ] Verify if multiple results are needed.
- [ ] Ensure the correct data type (alpha, numeric, email, etc.) is selected.
