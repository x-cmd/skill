---
name: gh
description: >
  Enhanced GitHub CLI for managing repositories, issues, PRs, actions, and GitHub Models.
  Core Scenario: When the user needs to automate GitHub workflows, manage secrets, or use GitHub's AI models via CLI.
license: MIT
---

# gh - GitHub Workflow Management

The `gh` module provides a comprehensive CLI for managing GitHub activities. It supports everything from basic repository operations to advanced features like GitHub Action artifacts and AI model interaction.

## When to Activate
- When managing GitHub repositories (clone, create, delete).
- When automating Issue and Pull Request (PR) lifecycles.
- When managing GitHub Actions, workflows, and CI/CD artifacts.
- When configuring secrets or managing organizational team memberships.
- When interacting with GitHub Models for AI-assisted development.

## Core Principles & Rules
- **Token Required**: Remind users to initialize their GitHub personal access token via `init`.
- **Interactive Apps**: Use `repo app` for a visual TUI to manage repositories.
- **AI Integration**: Leverage the `model` subcommand for GitHub's native AI capabilities.

## Patterns & Examples

### View User Repo (Interactive)
```bash
# Open an interactive TUI to browse your GitHub repositories
x gh repo app
```

### Manage PRs
```bash
# List all open pull requests for the current repository
x gh pr ls
```

### AI Models
```bash
# List available GitHub Models for AI tasks
x gh model ls
```

## Checklist
- [ ] Ensure the GitHub token is correctly initialized.
- [ ] Confirm if the operation is for a personal or organizational account.
- [ ] Verify the target repository name and owner.
