---
name: speckit
description: >
  Enhancement module for Spec Kit, helping users set up and use the Specify CLI for spec-driven development.
  Core Scenario: When the user wants to initialize a Specify project to formalize the "Requirement → Plan → Task → Implementation" flow.
license: MIT
---

# speckit - Spec Kit & Specify CLI Enhancement

The `speckit` module facilitates the use of Spec Kit and the Specify CLI, which guide developers and AI tools through a formalized, specification-driven development process to ensure high-quality, predictable outputs.

## When to Activate
- When the user wants to initialize a new Specify project to formalize development.
- When the user needs to set up a specific AI assistant (e.g., Claude) for a Specify project.
- When the user wants to check if all necessary tools for Spec Kit are installed.

## Core Principles & Rules
- **Formalized Workflow**: Emphasize the flow from requirements to implementation to eliminate uncertainty.
- **AI-Specific Initialization**: Support initializing projects with specific AI assistants like `claude` or `codex`.

## Additional Scenarios
- **Tool Validation**: Use `check` to ensure the environment is ready for spec-driven development.

## Patterns & Examples

### Initialize Project (Interactive)
```bash
# Set up a Specify project with interactive AI selection
x speckit init
```

### Initialize with Specific AI
```bash
# Set up a project in a specific directory with Claude as the assistant
x speckit init my-project --ai claude
```

## Checklist
- [ ] Confirm if the project needs a specific AI assistant during initialization.
- [ ] Run `x speckit check` to ensure all required tools are available.
