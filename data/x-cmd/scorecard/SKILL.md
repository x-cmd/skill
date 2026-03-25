---
name: scorecard
description: >
  Automated security tool for assessing open-source project risks and best practices adherence.
  Core Scenario: When the user needs to evaluate the security health of a GitHub repository or package.
license: MIT
---

# scorecard - OpenSSF Security Scorecard

The `scorecard` module evaluates open-source projects based on security best practices, providing a score and detailed report on potential risks like binary artifacts, unreviewed code, or dangerous workflows.

## When to Activate
- When the user wants to assess the security level of an open-source repository.
- When performing due diligence on a new dependency (npm, PyPI, etc.).
- When auditing a local repository for security improvements.

## Core Principles & Rules
- **Best Practices**: Focuses on identifying risks like lack of CI tests, missing branch protection, or pinned dependencies.
- **Detailed Reporting**: Use `--show-details` to understand why specific checks passed or failed.

## Patterns & Examples

### Repository Audit
```bash
# Display the security scorecard for a GitHub repository
x scorecard info github.com/ossf/scorecard
```

### Open Web Report
```bash
# Open the full OpenSSF scorecard report in a browser
x scorecard open github.com/owner/repo
```

## Checklist
- [ ] Confirm the target repository URL or package name.
- [ ] Verify if the user needs a summary or a detailed check breakdown.
