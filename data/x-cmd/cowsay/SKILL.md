---
name: cowsay
description: >
  Generate ASCII art pictures of cows or penguins with a custom message.
  Core Scenario: When the user wants to display messages in a fun, visual ASCII format in the terminal.
license: MIT
---

# cowsay - ASCII Art Messages

The `cowsay` module generates a visual ASCII representation of a cow (or other animals like a penguin) that "says" a provided message.

## When to Activate
- When the user wants to add humor or visual emphasis to terminal output.
- When combined with other tools (like `colr`) for creative terminal displays.

## Patterns & Examples

### Classic Cowsay
```bash
# Display a message with an ASCII cow
x cowsay "Hello world!"
```

### Penguin (Tux)
```bash
# Display a message with an ASCII penguin
x cowsay tux "Welcome to Linux"
```

## Checklist
- [ ] Confirm the message content.
- [ ] Verify if a specific animal (cow/tux) is requested.
