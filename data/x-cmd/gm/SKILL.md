---
name: gm
description: >
  Enhanced interface for GraphicsMagick, providing powerful image processing and conversion tools.
  Core Scenario: When the user needs to convert image formats, resize photos, or perform batch image editing.
license: MIT
---

# gm - GraphicsMagick Image Processor

The `gm` module provides an enhanced interface for GraphicsMagick, allowing users to process images efficiently. It automatically handles installation via pixi if the tool is not found locally.

## When to Activate
- When converting images between different formats (e.g., JPG to PNG).
- When resizing images or performing batch transformations.
- When comparing two images or creating image montages.
- When needing image identification and metadata descriptions.

## Core Principles & Rules
- **Tool Reliability**: Uses `pixi` to ensure GraphicsMagick is available across different environments.
- **Standard Commands**: Supports all classic `gm` subcommands like `convert`, `mogrify`, and `identify`.

## Patterns & Examples

### Convert Image
```bash
# Convert a JPG image to PNG format
x gm convert test.jpg test.png
```

### Resize Image
```bash
# Resize an image to 300px width
x gm convert -resize 300 test.jpg output.jpg
```

### List Formats
```bash
# View all supported image formats for conversion
x gm convert -list formats
```

## Checklist
- [ ] Confirm the target operation (convert, resize, etc.).
- [ ] Verify input file paths and formats.
