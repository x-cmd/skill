---
name: ffmpeg
description: >
  Powerful multimedia processing tool for converting, recording, and streaming audio and video.
  Core Scenario: When the user needs to convert media formats, extract audio, or perform complex video editing via CLI.
license: MIT
---

# ffmpeg - Comprehensive Multimedia Processor

The `ffmpeg` module provides a powerful CLI for processing audio, video, and other multimedia files. It ensures availability by automatically installing ffmpeg via pixi if it is missing from the system.

## When to Activate
- When converting between different video or audio formats (e.g., MP4 to AVI).
- When extracting audio tracks from video files (e.g., MP4 to MP3).
- When performing complex media operations like resizing, re-encoding, or applying filters.
- When needing to stream multimedia content from the terminal.

## Core Principles & Rules
- **Zero-Setup**: Automatically handles installation via `pixi` if necessary.
- **Pass-through**: Supports all standard `ffmpeg` arguments via the `--` or `cmd` subcommand.
- **Input/Output**: Always clarify the `-i` (input) and target output paths.

## Patterns & Examples

### Format Conversion
```bash
# Convert a video from mp4 to avi
x ffmpeg -i input.mp4 output.avi
```

### Extract Audio
```bash
# Extract audio from a video file and save as MP3
x ffmpeg -i input.mp4 -vn output.mp3
```

### Direct Command
```bash
# Execute a raw ffmpeg command string
x ffmpeg --cmd -version
```

## Checklist
- [ ] Confirm the input file exists and is accessible.
- [ ] Verify the desired output format and codec settings.
- [ ] Ensure appropriate disk space for large media operations.
