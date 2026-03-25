---
name: githook
description: >
  Manage Git hooks efficiently, supporting initialization, listing, and removal.
  Core Scenario: When the user needs to set up or audit Git hooks for automation and policy enforcement.
license: MIT
---

# githook - Git Hook Management Utility

The `githook` module provides a simple interface for managing Git hooks within a repository, enabling users to easily install, view, or remove hooks for tasks like linting or automated testing.

## When to Activate
- When setting up pre-commit or post-merge hooks in a repository.
- When auditing existing hooks to understand automated repository behaviors.

## Patterns & Examples

### List Hooks
```bash
# View all currently active hooks in the Git repository
x githook ls
```

## Checklist
- [ ] Confirm the repository path.
- [ ] Verify the specific hook type (pre-commit, etc.).
