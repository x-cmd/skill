---
name: shodan
description: >
  Search the Shodan network database for connected devices, vulnerabilities, and real-time network intelligence.
  Core Scenario: When the user needs to perform network reconnaissance, search for exposed assets, or audit IPs.
license: MIT
---

# shodan - Shodan Network Search CLI

The `shodan` module provides a comprehensive CLI for the Shodan search engine, enabling network reconnaissance, vulnerability detection, and monitoring of internet-connected assets.

## When to Activate
- When performing reconnaissance on specific IPs or domain names.
- When searching for connected devices with specific vulnerabilities (CVEs) or open ports.
- When monitoring network alerts for specific assets.
- When generating real-time network intelligence summaries via AI integration (`::`).

## Core Principles & Rules
- **API Key Management**: Remind users to obtain and initialize their Shodan API key via `init`.
- **Targeted Search**: Support for powerful filtering based on facets (ports, protocols, countries).
- **Data Export**: Can download host information and export to structured formats like CSV.

## Patterns & Examples

### Scan IPs
```bash
# Perform a targeted scan on specific IP addresses and ports
x shodan scan create 8.8.8.8 1.1.1.1=53/dns-udp,443/https
```

### Host Information
```bash
# Lookup IP information in the Shodan internet database
x shodan internetdb 8.8.8.8
```

### AI Intelligence
```bash
# Use AI to summarize Shodan results for specific queries
x shodan :: "critical vulnerabilities in industrial control systems"
```

## Checklist
- [ ] Ensure the Shodan API key is configured.
- [ ] Verify the search query or target IP range.
- [ ] Confirm if the user needs raw data or a summarized report.
