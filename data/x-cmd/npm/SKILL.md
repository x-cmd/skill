---
name: npm
description: >
  Enhanced interface for npm, the Node.js package manager.
  Core Scenario: When the user needs to manage JavaScript dependencies or run npm scripts.
license: MIT
---

# npm - Node.js Package Management

The `npm` module provides integrated access to the Node Package Manager, simplifying dependency installation and script execution for Node.js projects.

## When to Activate
- When installing or managing Node.js packages.
- When running predefined scripts in `package.json`.

## Patterns & Examples

### Install Global Package
```bash
# Install a package globally via npm
x npm install -g nodemon
```

## Checklist
- [ ] Confirm the package name.
- [ ] Verify the target project directory.
