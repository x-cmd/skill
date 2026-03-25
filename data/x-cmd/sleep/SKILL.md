---
name: sleep
description: >
  Pause execution for human-readable durations or schedule periodic tasks.
  Core Scenario: When the user needs to delay scripts, create timers (beacons), or schedule repeating commands.
license: MIT
---

# sleep - Pausing & Periodic Scheduling

The `sleep` module enhances the standard sleep command by supporting human-readable time formats and offering integrated scheduling for repeating tasks.

## When to Activate
- When a script needs to pause for a duration like `5m` or `1h30m`.
- When creating a periodic "beacon" that outputs timestamps.
- When scheduling a command to run at regular intervals (polling, backups, tests).

## Core Principles & Rules
- **Time Units**: Use `s`, `m`, `h`, `d` for intuitive duration specification.
- **Combined Command**: Pause then immediately execute a command using `x sleep <time> <command>`.
- **Scheduled Loops**: Use `schd` for controlled repeating tasks, with optional iteration limits (`-n`).

## Patterns & Examples

### Delay Script
```bash
# Pause for 1 minute then continue
x sleep 1m
```

### Periodic Timestamp (Beacon)
```bash
# Output current timestamp every 5 seconds
x sleep beacon 5s
```

### Repeating Task
```bash
# Run a backup script every hour, infinitely
x sleep schd -i 1h -- ./backup.sh
```

### Limited Repeating Task
```bash
# Run a test command every 10 seconds, for 5 iterations
x sleep schd -i 10s -n 5 -- make test
```

## Checklist
- [ ] Confirm the delay duration and its units.
- [ ] Verify if the task should repeat infinitely or a fixed number of times.
- [ ] Ensure the command to be scheduled is correctly provided after the `--` separator.
