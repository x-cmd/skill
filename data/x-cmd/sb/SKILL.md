---
name: sb
description: >
  SBOM (Software Bill of Materials) utility for generating and analyzing software component lists.
  Core Scenario: When the user needs to audit project dependencies or generate SBOM reports for security compliance.
license: MIT
---

# sb - SBOM Generator & Analyzer

The `sb` module provides tools for working with Software Bill of Materials (SBOM), enabling users to generate detailed lists of project components and perform security-focused analysis.

## When to Activate
- When generating SBOM reports for a project (e.g., using `syft` or `trivy` backends).
- When auditing software dependencies for security compliance.
- When converting between different SBOM formats.

## Core Principles & Rules
- **Automation**: Designed to integrate into CI/CD pipelines for automated SBOM generation.
- **Backend Support**: Leverages popular SBOM tools under the hood.

## Patterns & Examples

### Generate SBOM
```bash
# Create an SBOM report for the current project
x sb gen
```

### View Component List
```bash
# List all software components in a human-readable format
x sb ls
```

## Checklist
- [ ] Confirm the target project directory.
- [ ] Verify if a specific SBOM format (CycloneDX, SPDX) is required.
