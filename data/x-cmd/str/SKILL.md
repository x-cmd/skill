---
name: str
description: >
  High-performance string manipulation tool for transformations, splitting, and encoding.
  Core Scenario: When the user needs to process strings (upper/lower, trim, split, hash) in scripts or CLI.
license: MIT
---

# str - High-Performance String Manipulation

The `str` module is a low-level, high-performance tool for common string operations. It is essential for data processing within shell scripts, supporting both piping and direct arguments.

## When to Activate
- When converting string cases (upper/lower).
- When trimming whitespace or splitting strings by delimiters.
- When performing Base64 encoding/decoding or calculating MD5/SHA256 hashes.
- When replacing substrings or joining multiple strings.
- When converting line endings (dos2unix/unix2dos).

## Core Principles & Rules
- **Piping Support**: Highly optimized for use in terminal pipes (e.g., `echo "..." | x str upper`).
- **Python-Style Slicing**: Use the `slice` subcommand for complex substring extractions.
- **Efficiency**: Designed for minimal overhead in high-frequency script calls.

## Additional Scenarios
- **Hash Verification**: Quickly generate hashes for strings or data streams.
- **Format Conversion**: Standardize text files across different OS environments using `dos2unix`.

## Patterns & Examples

### Basic Transformation
```bash
# Convert a string to uppercase
x str upper "hello world"
```

### Splitting and Joining
```bash
# Split by comma and join by space
echo "a,b,c" | x str split "," | x str join " "
```

### Encoding and Hashing
```bash
# Base64 encode and generate SHA256 hash
x str base64 "secret"
x str sha256 "data"
```

## Checklist
- [ ] Confirm if the input is provided via pipe or argument.
- [ ] Verify the delimiter for splitting/joining tasks.
- [ ] Ensure the correct hash algorithm (MD5/SHA256) is requested.
