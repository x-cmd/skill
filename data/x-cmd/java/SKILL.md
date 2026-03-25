---
name: java
description: >
  Enhanced Java (JDK) environment manager for installing, switching, and running Java applications.
  Core Scenario: When the user needs to manage JDK versions or run Java/Jar files via CLI.
license: MIT
---

# java - Java (JDK) Environment Management

The `java` module provides a comprehensive way to manage Java Development Kits (JDK) and execute Java applications, ensuring that the appropriate runtime is available across different systems.

## When to Activate
- When installing or switching between multiple JDK versions.
- When running `.jar` files or compiling Java source code.
- When identifying the currently active Java environment.

## Patterns & Examples

### Run Jar File
```bash
# Execute a Java archive file
x java -jar myapp.jar
```

## Checklist
- [ ] Verify the desired JDK version.
- [ ] Ensure the `.jar` or `.java` file path is correct.
