---
name: mankier
description: >
  Command-line client for ManKier.com, offering manual page queries and detailed command explanation.
  Core Scenario: When the user needs to explain shell commands or fetch specific sections of online manual pages.
license: MIT
---

# mankier - Online Manual Page Queries & Explanation

The `mankier` module provides a CLI interface for the ManKier.com API, allowing users to search, browse, and explain manual pages directly from the web without local installation.

## When to Activate
- When the user needs a detailed breakdown of a shell command's flags and arguments.
- When searching for manual pages that are not installed on the local system.
- When retrieving specific sections (e.g., NAME, DESCRIPTION) of a manual page.
- When following cross-references between different manual pages.

## Core Principles & Rules
- **Remote Accuracy**: Uses the latest online documentation from ManKier.com.
- **Granular Retrieval**: Supports fetching specific sections using the `section` subcommand.
- **Command breakdown**: Prioritize the `explain` subcommand for translating technical flags into human-readable text.

## Patterns & Examples

### Command Breakdown
```bash
# Explain the meaning of flags in a jq command
x mankier explain jq -cr
```

### Fetch Section
```bash
# Retrieve only the NAME section of the tar manual
x mankier section NAME tar
```

### Web Search Integration
```bash
# Search ManKier via DuckDuckGo for NVMe related pages
x mankier : nvme
```

## Checklist
- [ ] Confirm the command or flag the user needs explained.
- [ ] Verify if a specific section of the manual is required.
- [ ] Ensure the user is aware the data is fetched from an online source.
