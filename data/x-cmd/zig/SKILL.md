---
name: zig
description: >
  Enhanced Zig language module for package management, project building, and ZON format conversion.
  Core Scenario: When the user needs to manage Zig projects, convert ZON to JSON/YAML, or initialize C compilation with Zig.
license: MIT
---

# zig - Zig Language Enhancement

The `zig` module extends the capabilities of the Zig toolchain, providing simplified package management, powerful format conversion for `build.zig.zon`, and integrated C compilation utilities.

## When to Activate
- When managing Zig packages and searching for common libraries (`pm`).
- When converting `build.zig.zon` files to JSON or YAML for analysis.
- When initializing C compilation using Zig as a drop-in replacement for standard compilers (`initcc`).
- When performing standard Zig tasks like building, formatting, or testing projects.

## Core Principles & Rules
- **Format Conversion**: Use the `zon` subcommand to transform Zig's internal format into machine-readable JSON/YAML.
- **C Integration**: Leverage `cc`, `c++`, and `ar` to use Zig's built-in toolchain for C projects.
- **Package Management**: Use `pm la` to discover frequently used Zig packages.

## Patterns & Examples

### Convert ZON to YAML
```bash
# Output the build.zig.zon content as YAML
cat build.zig.zon | x zig zon toyml
```

### Discover Packages
```bash
# List all available Zig packages in the x-cmd collection
x zig pm la
```

### Initialize C Compilation
```bash
# Setup Zig for C project compilation
x zig initcc
```

## Checklist
- [ ] Confirm if the user needs Zig-native tasks or C-integration utilities.
- [ ] Verify the target file paths for building or formatting.
- [ ] Check if JSON or YAML is preferred for ZON data conversion.
