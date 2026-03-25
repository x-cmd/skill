---
name: hi
description: >
  Display detailed HTTP information and network metadata.
  Core Scenario: When the user needs a quick summary of their current HTTP environment or request headers.
license: MIT
---

# hi - HTTP Information Summary

The `hi` module provides a quick way to view current HTTP metadata and request information, often used for debugging connectivity or viewing public IP info from an HTTP perspective.

## When to Activate
- When the user wants to see their public-facing HTTP headers and IP.
- When performing basic network diagnostic checks related to web requests.

## Patterns & Examples

### View HTTP Info
```bash
# Display general HTTP environment and connection details
x hi
```

## Checklist
- [ ] Confirm if the user needs public IP info or header details.
