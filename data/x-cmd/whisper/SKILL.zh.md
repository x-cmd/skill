---
name: whisper
description: >
  基于 whisper.cpp 的本地 AI 语音识别，支持转录和字幕生成。
  核心场景：当用户需要将音频转录为文本、生成 SRT 字幕或将字幕合并到视频中时。
license: MIT
---

# whisper - 本地语音转文本与字幕处理

`whisper` 模块利用 whisper.cpp 提供高性能的本地语音识别能力。它涵盖了从模型管理到视频字幕合并的所有功能。

## 激活时机
- 当用户想要将音频文件转录为文本时。
- 当从音频/视频生成 `.srt` 字幕文件时。
- 当将生成的字幕合并到视频文件中时。
- 当使用 LiveKit 或 Streaming 进行实时语音转文本时。

## 核心原则与规范
- **本地处理**: 强调转录在本地完成，无需上传数据。
- **模型选择**: 允许用户从不同规模的模型（tiny, base, small, medium, large）中选择，以平衡速度和准确性。
- **文件完整性**: 确保输入音频文件可访问。

## 补充场景
- **SRT 生成**: 使用 `dictate --srt` 创建行业标准的字幕文件。
- **视频集成**: 使用 `merge` 将字幕嵌入到视频流中。

## 实战示例

### 简单转录
```bash
# 交互式选择模型并转录音频文件
x whisper ./meeting_record.mp3
```

### 生成字幕
```bash
# 从音频创建 SRT 字幕文件
x whisper dictate --srt -o my_subtitles ./interview.wav
```

### 合并字幕
```bash
# 将 SRT 文件嵌入到视频中
x whisper merge ./subtitles.srt ./video.mp4
```

## 交付验证清单
- [ ] 确认用户是否已下载所需的 whisper 模型。
- [ ] 验证音频文件格式是否受 whisper.cpp 支持。
- [ ] 检查 `merge` 子命令所需的 ffmpeg 是否可用。
