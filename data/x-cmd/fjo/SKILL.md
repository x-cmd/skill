---
name: fjo
description: >
  CLI interface for Forgejo, the community-driven self-hosted software development platform.
  Core Scenario: When the user needs to manage repositories or issues on a Forgejo instance.
license: MIT
---

# fjo - Forgejo CLI Browser

The `fjo` module provides a dedicated CLI for Forgejo instances, supporting repository lifecycle management and collaboration features.

## When to Activate
- When working with Forgejo-based git hosting services.
- When managing issues or pull requests on Forgejo instances.

## Patterns & Examples

### List Issues
```bash
# View open issues for a specific Forgejo repository
x fjo issue ls owner/repo
```

## Checklist
- [ ] Confirm the Forgejo instance URL and credentials.
