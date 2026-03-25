---
name: npx
description: >
  Enhanced interface for npx, used to execute Node.js packages without local installation.
  Core Scenario: When the user needs to run a one-off Node.js tool or test a package.
license: MIT
---

# npx - Node Package Executor

The `npx` module allows users to execute Node.js tools directly from the npm registry without needing to install them globally or locally first.

## When to Activate
- When running a CLI tool once (e.g., scaffolding a project).
- When testing different versions of a package.

## Patterns & Examples

### Run Scaffolding Tool
```bash
# Execute create-react-app without installation
x npx create-react-app my-new-project
```

## Checklist
- [ ] Confirm the package name and arguments.
- [ ] Verify if a specific version is required.
