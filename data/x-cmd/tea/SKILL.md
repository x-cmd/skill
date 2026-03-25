---
name: tea
description: >
  Enhanced interface for the Gitea CLI (tea), enabling management of repositories, issues, and PRs.
  Core Scenario: When the user needs to manage projects on a self-hosted or public Gitea instance.
license: MIT
---

# tea - Gitea CLI Enhancement

The `tea` module provides a powerful CLI interface for Gitea, simplifying repository management, issue tracking, and pull request workflows.

## When to Activate
- When managing projects on Gitea servers.
- When automating development tasks like PR reviews or Issue creation on Gitea.

## Core Principles & Rules
- **Token Required**: Ensure the Gitea access token is initialized.
- **Environment Agnostic**: Works with both self-hosted and official Gitea instances.

## Patterns & Examples

### List Repositories
```bash
# View all repositories in the configured Gitea account
x tea repo ls
```

## Checklist
- [ ] Confirm the Gitea instance URL and access token.
