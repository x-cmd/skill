---
name: gddy
description: >
  Manage GoDaddy domains and DNS records from the CLI, supporting domain search and record modification.
  Core Scenario: When the user needs to audit their domain list, check availability, or edit DNS records.
license: MIT
---

# gddy - GoDaddy Domain & DNS Management

The `gddy` module provides a CLI interface for the GoDaddy API, enabling users to manage their domains and DNS records securely from the terminal.

## When to Activate
- When the user wants to list all domains in their GoDaddy account.
- When checking if a specific domain name is available for registration.
- When adding, removing, or viewing DNS records (A, CNAME, etc.) for a domain.
- When managing GoDaddy API keys and configurations.

## Core Principles & Rules
- **API Credentials**: Remind users to configure their API key and secret via `init` or `--cfg`.
- **Domain Search**: If a subcommand is not recognized, the module automatically treats it as a domain availability search.
- **Destructive Edits**: Use caution when removing DNS records.

## Patterns & Examples

### List Domains
```bash
# View all domains in the current account
x gddy domain ls
```

### Modify DNS Record
```bash
# Add a new DNS record to a domain
x gddy domain record add --name dev --data "1.2.3.4" my-domain.com
```

### Domain Availability
```bash
# Check if a specific domain is available
x gddy search example.com
```

## Checklist
- [ ] Ensure GoDaddy API key and secret are properly initialized.
- [ ] Confirm the target domain name and record details.
- [ ] Verify if the production or sandbox environment is intended.
