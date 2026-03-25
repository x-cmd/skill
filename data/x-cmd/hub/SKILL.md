---
name: hub
description: >
  Securely upload, host, and share scripts via x-cmd cloud, supporting AI-driven knowledge base search.
  Core Scenario: When the user needs to manage cloud-hosted scripts or search the x-cmd knowledge base using AI.
license: MIT
---

# hub - Cloud Script Management & AI Search

The `hub` module provides a secure space for managing and sharing scripts in the x-cmd cloud. It supports encrypted hosting, cross-device access, and an AI-powered search for the x-cmd knowledge base.

## When to Activate
- When the user wants to upload a local script to the cloud for sharing or cross-device use.
- When performing an AI search within the x-cmd knowledge base for specific module usage.
- When compiling Go, Gop, or Zig scripts in a remote cloud environment.
- When managing x-cmd hub account and file permissions.

## Core Principles & Rules
- **Security**: Supports encrypted hosting to protect script privacy.
- **Knowledge Retrieval**: Use the `search` or `ai` subcommands to leverage the AI knowledge base.
- **Access Control**: Use `access` to toggle between public and private file states.

## Patterns & Examples

### Upload and Share
```bash
# Upload a file and set it to public
x hub file put --public my_script.sh
```

### AI Search
```bash
# Search for how to add members in a GitHub repo via x-cmd
x hub search "how to add members in github repo"
```

### Account Info
```bash
# Check current cloud space usage and account details
x hub info
```

## Checklist
- [ ] Ensure the user is logged in via `x hub login`.
- [ ] Confirm if the file should be public or private.
- [ ] Verify if the AI search query is specific enough.
