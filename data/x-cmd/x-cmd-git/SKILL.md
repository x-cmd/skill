---
name: x-cmd-git
tag: git
description: This skill provides comprehensive Git and code hosting platform management tools through x-cmd CLI, including GitHub, GitLab, Codeberg, Forgejo integration, and Git hooks management. This skill should be used when users need to manage Git repositories, work with code hosting platforms, automate Git workflows, or configure Git hooks from command line interfaces.
---

# x-cmd Git and Code Hosting Tools

## Overview

This skill provides professional Git and code hosting platform management capabilities through the x-cmd ecosystem. The tools enable developers, DevOps engineers, and open source contributors to manage repositories, collaborate on code, automate workflows, and integrate with multiple Git hosting services directly from the terminal.

## Available Tools

### GitHub Management (gh)
Comprehensive GitHub platform integration and management.

- **Usage**: `x gh [subcommand]`
- **Key subcommands**:
  - `repo` - Repository management
  - `issue` - Issue tracking and management
  - `pr` - Pull request management
  - `action` - GitHub Actions workflow management
  - `user` - User profile and account management
  - `search` - Repository and topic search
  - `browse` - Open GitHub resources in browser
- **Examples**:
  - `x gh user info` - Get current user information
  - `x gh repo app` - Interactive repository viewer
  - `x gh repo clone owner/repo` - Clone repository
  - `x gh pr create` - Create pull request
  - `x gh action workflow` - Manage workflows

### GitLab Management (gl)
Complete GitLab platform integration and administration.

- **Usage**: `x gl [subcommand]`
- **Key subcommands**:
  - `repo` - Repository management
  - `issue` - Issue management
  - `mr` - Merge request management
  - `user` - User administration
  - `group` - Group and team management
  - `deploy` - Deployment management
  - `snippet` - Code snippet management
- **Examples**:
  - `x gl repo ls` - List repositories
  - `x gl mr create` - Create merge request
  - `x gl user info` - Get user information
  - `x gl group ls` - List groups
  - `x gl repo clone project` - Clone repository

### Codeberg Management (cb)
Lightweight CLI for Codeberg open source hosting.

- **Usage**: `x cb [subcommand]`
- **Key subcommands**:
  - `repo` - Repository management
  - `user` - User profile management
  - `org` - Organization administration
  - `issue` - Issue tracking
  - `pr` - Pull request management
  - `notification` - Notification handling
- **Examples**:
  - `x cb repo ls` - List repositories
  - `x cb user info` - Get user information
  - `x cb issue create` - Create issue
  - `x cb pr list` - List pull requests
  - `x cb repo clone owner/repo` - Clone repository

### Forgejo Management (fjo)
Self-hosted Git platform management for Forgejo instances.

- **Usage**: `x fjo [subcommand]`
- **Key subcommands**:
  - `repo` - Repository management
  - `user` - User administration
  - `org` - Organization management
  - `issue` - Issue tracking
  - `notification` - Notification handling
- **Examples**:
  - `x fjo repo ls` - List repositories
  - `x fjo user info` - Get user information
  - `x fjo issue create` - Create issue
  - `x fjo pr create` - Create pull request
  - `x fjo repo clone project` - Clone repository

### Git Hooks Management (githook)
Git hooks configuration and automation.

- **Usage**: `x githook [subcommand]`
- **Key subcommands**:
  - `apply` - Apply Git hooks configuration
  - `clear` - Clear hooks and remove configuration
- **Examples**:
  - `x githook apply` - Apply hooks from configuration
  - `x githook clear` - Remove all hooks configuration

## Git and Code Hosting Use Cases

### Repository Management
- Use `x gh repo` for GitHub repository operations
- Use `x gl repo` for GitLab repository management
- Use `x cb repo` for Codeberg repository handling
- Use `x fjo repo` for Forgejo repository administration

### Collaboration and Code Review
- Use `x gh pr` for GitHub pull request workflows
- Use `x gl mr` for GitLab merge request processes
- Use `x cb pr` for Codeberg pull request management
- Use `x fjo pr` for Forgejo pull request handling

### Issue Tracking and Project Management
- Use `x gh issue` for GitHub issue management
- Use `x gl issue` for GitLab issue tracking
- Use `x cb issue` for Codeberg issue handling
- Use `x fjo issue` for Forgejo issue management

### CI/CD and Automation
- Use `x gh action` for GitHub Actions workflows
- Use `x gl` deployment features for GitLab CI/CD
- Use `x githook` for local Git automation
- Use platform-specific automation features

### User and Team Administration
- Use `x gh user` for GitHub user management
- Use `x gl user` and `x gl group` for GitLab administration
- Use `x cb user` and `x cb org` for Codeberg organization
- Use `x fjo user` and `x fjo org` for Forgejo administration

## Installation and Setup

### Prerequisites
- x-cmd CLI installed
- Git installed and configured
- Internet connectivity for platform operations

### Platform Authentication

#### GitHub Setup
```bash
x gh init  # Interactive configuration
x gh --cfg token=<github-token>
```
Get GitHub token from: https://github.com/settings/tokens

#### GitLab Setup
```bash
x gl init  # Interactive configuration
x gl --cfg token=<gitlab-token>
```
Get GitLab token from: https://gitlab.com/-/profile/personal_access_tokens

#### Codeberg Setup
```bash
x cb init  # Interactive configuration
x cb --cfg token=<codeberg-token>
```
Get Codeberg token from: https://codeberg.org/user/settings/applications

#### Forgejo Setup
```bash
x fjo init  # Interactive configuration
x fjo --cfg token=<forgejo-token>
```
Configure Forgejo instance and token

### Git Hooks Configuration
```bash
# Apply hooks configuration
x githook apply

# Clear hooks configuration
x githook clear
```

## Integration with Other Tools

### AI and Code Assistance
- Use `--co` flag for AI code copilot functionality
- Use `ddgoai` for AI-powered search and summarization
- Integrate with other AI tools for code generation

### Data Processing and Analysis
- Pipe output to `x jq` for JSON processing
- Use with `@zh` for Chinese translation
- Export to CSV/TSV for reporting and analysis

### Development Workflows
- Combine with `x curl` for API interactions
- Use with shell scripts for automation
- Integrate with CI/CD pipelines

## Troubleshooting

### Common Issues
- **Authentication errors**: Verify API tokens and permissions
- **Network connectivity**: Check internet connection for platform operations
- **Permission issues**: Ensure appropriate repository access rights
- **Configuration problems**: Verify platform-specific settings

### Performance Optimization
- Use specific queries rather than broad searches
- Cache authentication tokens securely
- Use interactive interfaces for complex operations
- Limit API calls to respect rate limits

### Security Best Practices
- **Token security**: Store API tokens securely and rotate regularly
- **Access control**: Follow principle of least privilege for repository access
- **Audit logging**: Maintain records of platform operations
- **Compliance**: Ensure usage complies with platform terms of service

## Support and Resources

- **x-cmd Git Documentation**: https://x-cmd.com/mod/git
- **GitHub CLI**: https://x-cmd.com/mod/gh
- **GitLab CLI**: https://x-cmd.com/mod/gl
- **Codeberg CLI**: https://x-cmd.com/mod/cb
- **Forgejo CLI**: https://x-cmd.com/mod/fjo
- **Git Hooks**: https://x-cmd.com/mod/githook

For additional help:
- Use `x [tool] --help` for specific tool documentation
- Visit individual module pages for detailed usage
- Check platform-specific API documentation
- Consult Git and DevOps best practices