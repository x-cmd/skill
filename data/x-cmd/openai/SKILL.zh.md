---
name: openai
description: >
  集成 OpenAI API，提供包括 ChatGPT 对话、文本生成、图片创作及语音转换在内的 AI 能力。
  核心场景：当用户需要与 OpenAI 模型交互进行聊天、翻译、图片生成或语音处理时。
license: MIT
---

# openai - OpenAI API 集成

`openai` 模块提供 OpenAI 服务的完整命令行界面，使用户能够直接从终端执行复杂的 AI 任务，如文本生成、翻译、图片创建及语音转换。

## 激活时机
- 当用户想要使用 OpenAI 进行聊天或文本生成时（例如，生成提交信息）。
- 当用户需要使用 AI 翻译文件内容时。
- 当用户想要根据文本提示生成图片时。
- 当用户需要将文本转换为语音或转录音频时。
- 当管理 OpenAI API 密钥或模型配置时。

## 核心原则与规范
- **API 密钥管理**: 使用 `init` 或 `--cfg apikey=<key>` 设置环境。
- **文件输入**: 使用 `--file` 标志向聊天/生成子命令提供来自本地文件的上下文。
- **重现性**: 请注意 AI 输出可能会有所不同；如果需要，请使用适当的参数以获得一致的结果。

## 补充场景
- **Git 提交信息**: 将 `git diff` 传送到 `@gpt` 以生成标准化的提交信息。
- **音频处理**: 将 `audio` 子命令用于 TTS（文本转语音）或转录任务。
- **嵌入和微调**: 高级用户可以管理微调模型或计算文本嵌入。

## 实战示例

### 带有文件上下文的聊天
```bash
# 使用 OpenAI 将多个文件翻译为中文
x openai chat request --file ./abstract.en.md --file ./content.en.md "翻译为中文"
```

### 图片生成
```bash
# 根据提示生成图片
x openai image create --prompt "未来城市的的高质量数字艺术"
```

### 文本转语音
```bash
# 将文本转换为音频文件
x openai audio generate --input "欢迎来到 x-cmd" --model tts-1 --voice alloy
```

### 生成提交信息
```bash
# 将 diff 传送到 gpt 以生成提交信息
git diff | @gpt "生成符合 Conventional Commits 格式的合适 Git 提交信息"
```

## 交付验证清单
- [ ] 确保已使用 `x openai init` 配置了 OpenAI API 密钥。
- [ ] 使用 `--file` 标志时确认输入文件存在。
- [ ] 确认语音生成的所需模型和声音。
