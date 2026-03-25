---
name: moonshot
description: >
  集成 Moonshot AI (Kimi)，用于聊天和基于文件的上下文处理。
  核心场景：当用户想要利用 Moonshot 的大上下文能力进行聊天或文件翻译时。
license: MIT
---

# moonshot - Moonshot AI (Kimi) 集成

`moonshot` 模块提供对 Moonshot AI 服务的访问，该公司以其大上下文窗口和在中文任务中的强劲表现而闻名。

## 激活时机
- 当用户想要与 Kimi 模型聊天时。
- 当用户需要利用 Moonshot 的上下文分析或翻译文件时。
- 当检查 Moonshot 账户余额或管理文件上传时。

## 核心原则与规范
- **API 密钥管理**: 使用 `init` 或 `--cfg apikey=<key>` 进行设置。
- **别名支持**: 使用 `@kimi` 别名进行更快速的访问。
- **上下文处理**: 通过使用 `--file` 附加文件来利用其大上下文窗口。

## 补充场景
- **文件管理**: 使用 `file` 子命令管理上传到 Moonshot 的文档。
- **余额检查**: 使用 `x moonshot balance` 监控信用额度。

## 实战示例

### 使用 Kimi 翻译文件
```bash
# 使用 @kimi 别名翻译本地文件
@kimi --file ./content.en.md "翻译为中文"
```

### 列出可用模型
```bash
# 查看 Moonshot 支持的模型
x moonshot model ls
```

## 交付验证清单
- [ ] 确保已配置 Moonshot API 密钥。
- [ ] 验证附加的文件是否存在且可读。
