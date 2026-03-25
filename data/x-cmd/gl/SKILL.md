---
name: gl
description: >
  Enhanced GitLab CLI for managing repositories, issues, snippets, and CI/CD deployments.
  Core Scenario: When the user needs to automate GitLab project management or coordinate deployments.
license: MIT
---

# gl - GitLab Workflow Management

The `gl` module provides an interface for GitLab project management, including support for snippets, deployment tracking, and group/subgroup coordination.

## When to Activate
- When managing repositories and teams on a GitLab instance.
- When coordinating CI/CD deployments via GitLab subcommands.
- When creating or managing project code snippets.

## Core Principles & Rules
- **Token Required**: Remind users to initialize their GitLab access token.
- **Broad Support**: Covers groups, subgroups, and individual project settings.

## Patterns & Examples

### Clone Repository
```bash
# Clone a specific GitLab repository using the shortcut
x gl cl owner/repo
```

### View Project Snippets
```bash
# List all code snippets associated with a project
x gl snippet ls
```

## Checklist
- [ ] Ensure the GitLab token is initialized.
- [ ] Verify if the command is targeting a self-hosted or cloud-based GitLab.
