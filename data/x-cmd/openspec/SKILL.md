---
name: openspec
description: >
  Lightweight specification-driven framework to coordinate human and AI coding collaboration.
  Core Scenario: When the user wants to initialize a project, manage change proposals, or validate project specs with AI.
license: MIT
---

# openspec - Spec-Driven Collaboration Framework

The `openspec` module provides a CLI interface for OpenSpec, a framework designed to streamline human-AI collaboration through a standardized, specification-driven development process.

## When to Activate
- When the user wants to initialize a new OpenSpec-driven project structure.
- When the user needs to manage change proposals (RFCs) or project specifications.
- When the user wants to validate consistency between project specifications and proposed changes.
- When the user needs an interactive dashboard to view the project's specs and progress.
- When updating the OpenSpec instruction files to better align AI assistants.

## Core Principles & Rules
- **Spec-First**: Follow the flow of defining specifications (`spec`) before proposing changes (`change`).
- **Standardized Instructions**: Use `update` to ensure AI assistants have the latest alignment instructions.
- **Project Structure**: OpenSpec relies on specific `.md` files to track state; use `init` to set up the environment.

## Additional Scenarios
- **Dashboard Overview**: Use `view` for an interactive summary of specs and changes.
- **Finalizing Changes**: Use `archive` to merge completed changes back into the main specifications.

## Patterns & Examples

### Initialize Project
```bash
# Set up OpenSpec structure in the current directory
x openspec init
```

### Create Change Proposal
```bash
# Start a new change proposal
x openspec change new
```

### Validate Alignment
```bash
# Check if current changes align with project specifications
x openspec validate
```

## Checklist
- [ ] Verify the project is initialized with `x openspec init`.
- [ ] Ensure AI assistant instructions are up-to-date by running `update`.
