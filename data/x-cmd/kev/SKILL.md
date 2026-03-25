---
name: kev
description: >
  Retrieve and list Known Exploited Vulnerabilities (KEV) from official security catalogs.
  Core Scenario: When the user needs to audit system safety against actively exploited security flaws.
license: MIT
---

# kev - Known Exploited Vulnerabilities Catalog

The `kev` module provides a simple interface to list and query known exploited vulnerabilities, helping security professionals prioritize patching based on real-world exploit data.

## When to Activate
- When performing a rapid audit of current security threats.
- When listing the top or recent vulnerabilities that are known to be active in the wild.

## Core Principles & Rules
- **Direct Retrieval**: Fetches data from authoritative security catalogs.
- **Filtering**: Supports viewing the full list or just the top N most critical/recent entries.

## Patterns & Examples

### List All KEVs
```bash
# Display the entire catalog of known exploited vulnerabilities
x kev ls
```

### Top Critical KEVs
```bash
# View the top 100 entries from the KEV list
x kev top 100
```

## Checklist
- [ ] Confirm if the user needs the full list or a limited summary.
