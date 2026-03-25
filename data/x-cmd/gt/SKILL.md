---
name: gt
description: >
  Enhanced Gitee CLI for managing repositories, issues, and PRs on the Gitee platform.
  Core Scenario: When the user needs to automate development workflows or manage projects on Gitee.
license: MIT
---

# gt - Gitee Workflow Management

The `gt` module provides an interface for managing Gitee activities, supporting repository lifecycle management, issue tracking, and pull request coordination.

## When to Activate
- When managing projects hosted on the Gitee platform.
- When automating PR reviews or Issue tracking within a Gitee-based team.
- When managing Gitee organizations or enterprise-level settings.

## Core Principles & Rules
- **Token Required**: Use `init` or `--cfg` to set the Gitee personal access token.
- **Shortcut Support**: Use `cl` as a shortcut for `repo clone`.

## Patterns & Examples

### View User Profile
```bash
# Retrieve current user information from Gitee
x gt user info
```

### List Repositories (Interactive)
```bash
# View and browse your Gitee repositories in a TUI table
x gt repo ls
```

## Checklist
- [ ] Confirm the Gitee token is configured.
- [ ] Verify if the project is part of a personal or enterprise account.
