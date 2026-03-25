---
name: osv
description: >
  CLI for Google's Open Source Vulnerabilities (OSV) project, providing vulnerability scanning and dependency analysis.
  Core Scenario: When the user needs to scan projects for vulnerabilities or query specific CVE/OSV IDs.
license: MIT
---

# osv - Open Source Vulnerabilities Scanner

The `osv` module provides an interface for the OSV project, enabling users to scan local projects, identify vulnerable dependencies, and retrieve detailed vulnerability information.

## When to Activate
- When the user wants to perform a security audit on their project dependencies (npm, pip, etc.).
- When querying detailed information for a specific vulnerability ID (e.g., `osv-2020-111`).
- When generating security reports in the SARIF format for integration with CI/CD pipelines.
- When searching for vulnerabilities related to specific software packages and versions.

## Core Principles & Rules
- **Comprehensive Scanning**: Use `sarif` to generate standardized security reports.
- **Eco-System Aware**: Supports multiple ecosystems including npm, pypi, and more.
- **Search Integration**: Uses AI or DuckDuckGo to summarize vulnerability details from the web.

## Patterns & Examples

### Full Project Scan
```bash
# Scan dependencies and generate a SARIF report
x osv sarif
```

### Specific Vulnerability Query
```bash
# Get details for a specific vulnerability ID
x osv vuln OSV-2020-111
```

### Check Software Version
```bash
# Query vulnerabilities for a specific version of a package
x osv q -p jq -v 1.7.1
```

## Checklist
- [ ] Confirm if the user needs a full scan or info on a specific ID.
- [ ] Verify the target project directory or package name.
- [ ] Ensure the correct ecosystem (pip, npm) is identified if using granular subcommands.
