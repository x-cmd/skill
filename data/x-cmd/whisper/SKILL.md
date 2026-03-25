---
name: whisper
description: >
  Local AI speech recognition based on whisper.cpp, supporting transcription and subtitle generation.
  Core Scenario: When the user needs to transcribe audio to text, generate SRT subtitles, or merge subtitles into video.
license: MIT
---

# whisper - Local Speech-to-Text & Subtitles

The `whisper` module provides a high-performance local speech recognition capability using whisper.cpp. It handles everything from model management to video subtitle merging.

## When to Activate
- When the user wants to transcribe an audio file into text.
- When generating `.srt` subtitle files from audio/video.
- When merging generated subtitles into a video file.
- When performing real-time speech-to-text using LiveKit or Streaming.

## Core Principles & Rules
- **Local Processing**: Emphasize that transcription happens locally without uploading data.
- **Model Selection**: Allow users to choose from different model sizes (tiny, base, small, medium, large) for speed vs. accuracy.
- **File Integrity**: Ensure input audio files are accessible.

## Additional Scenarios
- **SRT Generation**: Use `dictate --srt` to create industry-standard subtitle files.
- **Video Integration**: Use `merge` to embed subtitles into a video stream.

## Patterns & Examples

### Simple Transcription
```bash
# Interactively choose a model and transcribe an audio file
x whisper ./meeting_record.mp3
```

### Generate Subtitles
```bash
# Create an SRT subtitle file from audio
x whisper dictate --srt -o my_subtitles ./interview.wav
```

### Merge Subtitles
```bash
# Embed an SRT file into a video
x whisper merge ./subtitles.srt ./video.mp4
```

## Checklist
- [ ] Confirm if the user has downloaded the required whisper model.
- [ ] Verify the audio file format is supported by whisper.cpp.
- [ ] Check if ffmpeg is available for the `merge` subcommand.
