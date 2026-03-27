---
name: x-is
description: >
  Value validation and environment detection tool for Shell scripts. Validates input types (integers, floats, IPs),
  compares file ages, detects runtime environments (TTY, WSL, Termux, Cygwin). Supports batch checking with exit code 0 success / 1 failure.
  Trigger when user needs to validate input types, detect terminal environment, check file freshness, verify variable states, or perform environment suitability checks (suitable_pkg/suitable_advise).
  Keywords: validate, check, verify, is, integer, float, positive, negative, range, IP address, file age, interactive, WSL, Termux, environment detection, unset variable, batch check.
license: MIT
compatibility: POSIX Shell (sh/bash/zsh/dash/ash)

metadata:
  author: X-CMD
  version: "1.0.0"
  category: core
  tags: [shell, cli, tools, validation, type-check, environment-detection]
  repository: https://github.com/x-cmd/skill
  display_name: Value & Environment Validator
---

# x is - Value & Environment Validation

`x is` is x-cmd's value validation tool for checking if values or environment states meet specific criteria. Supports batch checking multiple values, ideal for robust Shell scripting.

## Prerequisites

1. Load x-cmd before use:
   ```bash
   . ~/.x-cmd.root/X
   ```

## Core Usage

- **Type Check**: `x is int 42 100 -5` / `x is float 3.14`
- **Range Check**: `x is minmax 1 100 50 75`
- **Compare**: `x is eq "a" "a"` / `x is within "ok" pending success`
- **IP Check**: `x is ip 192.168.1.1`
- **File Age**: `x is newest target src/*.c` / `x is oldest cache.txt /tmp/*.tmp`
- **Variable Check**: `x is unset VAR1 VAR2`
- **Environment Detection**: `x is interactive` / `x is wsl` / `x is termux`

## Subcommands

### Type
| Command | Description |
|---------|-------------|
| `int` | Integer check, batch supported |
| `float` | Float check (must contain decimal point) |
| `positive` | Positive integer (≥0) |
| `negative` | Negative integer (<0) |
| `minmax` | Range check `[MIN,MAX]` |

### Compare
| Command | Description |
|---------|-------------|
| `eq` | Multiple values equal |
| `within` | First value in candidate list |
| `in` | Colon-separated list member check |
| `in-` | Dash-separated list |
| `in_` | Underscore-separated list |
| `in/` | Slash-separated list |
| `in\|` | Pipe-separated list |

### Network
| Command | Description |
|---------|-------------|
| `ip` | IPv4 address check |

### File
| Command | Description |
|---------|-------------|
| `newest` | File newer than others (glob supported) |
| `oldest` | File older than others (glob supported) |

### Variable
| Command | Description |
|---------|-------------|
| `unset` | Variable is unset |

### Environment
| Command | Description |
|---------|-------------|
| `interactive` | Interactive session |
| `interactive_tty` | stdin is interactive TTY |
| `interactiveshell` | Shell is interactive |
| `repl` | REPL mode |
| `stdout2tty` | stdout outputs to TTY |
| `wsl` | WSL environment |
| `cygwin` | Cygwin environment |
| `msys` | MSYS/MinGW environment |
| `gitbash` | Git Bash environment |
| `termux` | Termux environment |
| `ish` | iSH environment |

### Suitability
| Command | Description |
|---------|-------------|
| `suitable_pkg` | Suitable for package installation |
| `suitable_advise_env` | Suitable for advise feature |
| `suitable_advise_repl` | Suitable for advise REPL |

## Practical Examples

### Batch Type Validation
```bash
x is int 1 2 3 4 5 && echo "All integers"
x is float 3.14 -2.5 .5 && echo "All floats"
```

### Environment Conditional Branch
```bash
x is interactive && echo "Interactive session" || echo "Script mode"
x is wsl && echo "WSL environment" || echo "Not WSL"
x is termux && echo "Termux environment"
```

### File Status Check
```bash
x is newest build.tar.gz src/*.c lib/*.h || echo "Rebuild needed"
```

### Variable Cleanup Verification
```bash
x is unset TEMP_VAR1 TEMP_VAR2 && echo "Temp variables cleaned"
```

### Script Pre-flight Check
```bash
x is suitable_pkg || { echo "Environment not suitable"; exit 1; }
```

## Design Principles

- **Batch First**: Check multiple values at once, all must pass
- **Exit Code Semantics**: 0 = success, 1 = failure, perfect for `&&`/`||` chains
- **Silent**: Validation tool outputs nothing, only exit code indicates result

## Get Help

```bash
x is --help
```
