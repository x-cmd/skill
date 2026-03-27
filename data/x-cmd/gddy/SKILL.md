---
name: gddy
description: >
  GoDaddy domain management tool.
  Core Scenario: When AI needs to check domain availability, list owned domains, or update DNS records.
license: MIT
---

# x gddy - GoDaddy Domain Management (AI Optimized)

The `x gddy` module allows managing your GoDaddy domains via the command line. It's ideal for DNS maintenance and domain querying within AI workflows.

## When to Activate
- When checking if a specific domain is available for purchase.
- When listing all domains owned by the GoDaddy account.
- When dynamically updating DNS resolution records (e.g., A records, CNAME).

## Core Principles & Rules
- **Non-interactive First**: Use subcommands directly, avoiding unnecessary UI confirmations.
- **Environment Requirements**: Requires API Key and Secret. If not configured, AI should guide the user through initialization.
- **Configuration Guidance**:
  - Direct the user to the GoDaddy Developer portal to get API info.
  - Suggest the user run `x gddy init` for configuration.

## Patterns & Examples

### Check Domain Availability
```bash
# Check if example.com is available for purchase
x gddy search example.com
```

### List Owned Domains
```bash
# List all domains under the account (non-interactive)
x gddy domain ls
```

### Manage DNS Records
```bash
# View detailed info for a specific domain
x gddy domain info example.com

# Add an A record for a domain
x gddy domain record add --name "www" --data "1.2.3.4" example.com
```

## Configuration Guide (for AI)
If an API error occurs or configuration is missing, provide this guidance to the user:
> Please obtain your API Key and Secret from the GoDaddy Developer portal (https://developer.godaddy.com/keys), then run the following command in your terminal to initialize:
> `x gddy init`

## Checklist
- [ ] Confirm if API credentials are valid.
- [ ] Confirm before modifying DNS records or performing purchases.
