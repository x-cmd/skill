---
name: pandoc
description: >
  Universal document converter supporting conversions between Markdown, HTML, PDF, and more.
  Core Scenario: When the user needs to convert document formats or fetch web content as Markdown.
license: MIT
---

# pandoc - Universal Document Converter

The `pandoc` module provides an interface for the powerful Pandoc document converter. It simplifies the installation process and enables easy conversion between diverse text and document formats.

## When to Activate
- When converting Markdown files to HTML, PDF, or Word documents.
- When fetching and converting webpage content into clean Markdown.
- When listing supported document formats or extensions.
- When needing a zero-setup document conversion tool.

## Core Principles & Rules
- **Zero-Setup**: Automatically downloads and manages the pandoc binary if it is missing.
- **Versatility**: Emphasize its ability to handle "anything to anything" conversions.
- **Standalone Support**: Encourage using the `-s` flag for full documents with headers/footers.

## Patterns & Examples

### Markdown to HTML
```bash
# Convert a Markdown file to a standalone HTML page
x pandoc -s input.md -o output.html
```

### Web to Markdown
```bash
# Convert a live webpage into a Markdown document
x pandoc -s -r html https://example.com -o webpage.md
```

### List Formats
```bash
# View all supported input document formats
x pandoc --list-input-formats
```

## Checklist
- [ ] Confirm the source and target document formats.
- [ ] Verify if a standalone document (`-s`) or a fragment is needed.
- [ ] Ensure input URLs or file paths are correct.
