---
name: node
description: >
  Enhanced Node.js module for version management, package installation (npm/npx), and environment setup.
  Core Scenario: When the user needs to manage Node.js versions, run JS scripts, or install npm packages.
license: MIT
---

# node - Node.js Development Environment

The `node` module simplifies Node.js development by offering integrated version management and direct access to npm and npx tools across various platforms.

## When to Activate
- When installing or switching Node.js versions.
- When using `npm` to manage dependencies or `npx` to run remote tools.
- When executing JavaScript files in the terminal.

## Patterns & Examples

### Install NPM Packages
```bash
# Install a Node package globally
x node npm install -g typescript
```

### Run Tool via NPX
```bash
# Execute a tool without local installation
x node npx create-react-app my-app
```

## Checklist
- [ ] Verify the desired Node.js version.
- [ ] Ensure npm/npx is required for the specific task.
