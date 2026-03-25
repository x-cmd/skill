---
name: writer
description: >
  AI text processing module for simplified translation, summarization, and refinement.
  Core Scenario: When the user needs to translate, polish, or summarize text using pre-configured AI agents.
license: MIT
---

# writer - AI Text Processing Assistant

The `writer` module provides high-level abstractions for common text processing tasks, allowing users to leverage AI for translation, summarization, and content optimization via simple subcommands.

## When to Activate
- When the user wants to translate text to Chinese (using `@zh`) or other languages.
- When generating summaries (`abs`) or refining existing text (`refine`).
- When expanding content or polishing writing for a better tone.
- When managing text processing agents and their default styles.

## Core Principles & Rules
- **Simplicity**: Use short aliases like `@zh` for the most common tasks.
- **Piping Support**: Designed to process text from `stdin` or local files.
- **Customization**: Use `init` to configure the preferred model and default writing style.

## Additional Scenarios
- **Spanish Translation**: Use the `tran` subcommand with specific targets for non-Chinese translations.
- **Code/Text Explanation**: Use `explain` to get a breakdown of complex paragraphs or scripts.

## Patterns & Examples

### Translate to Chinese (Short)
```bash
# Translate a string using the quick alias
@zh "The future of CLI is AI-driven"
```

### Summarize a File
```bash
# Generate a summary from a local text file
cat ./article.txt | @zh abs
```

### Refine and Polish
```bash
# Improve the flow and friendly tone of a text
x writer polish -f ./draft.md
```

## Checklist
- [ ] Ensure the AI processing agent is initialized via `x writer init`.
- [ ] Confirm if the output should be in a specific language (default is often Chinese).
- [ ] Verify the source text is correctly piped or pointed to via the `-f` flag.
