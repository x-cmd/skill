---
name: claude
description: >
  Claude Code 命令行增强模块，支持第三方模型提供商接入和会话管理。
  核心场景：当用户想要配合其他模型（如 DeepSeek）使用 Claude Code、统计 Token 消耗或管理 MCP 时。
license: MIT
---

# claude - Claude Code 增强模块

`claude` 模块扩展了 Anthropic 的 `claude-code` 能力，允许用户连接多种 AI 模型提供商、管理会话并监控 Token 消耗。

## 激活时机
- 当用户想要使用特定提供商（如 DeepSeek, MiniMax, 智谱）启动 Claude Code 时。
- 当用户需要全局或按项目配置 Claude Code 使用第三方模型时。
- 当用户想要查看过去 7 天的 Token 消耗和成本统计时。
- 当用户需要管理模型上下文协议 (MCP) 服务器时。
- 当用户想要移除或自定义 Claude Code 的作者署名或状态栏时。

## 核心原则与规范
- **连接提供商**: 使用 `ds` (DeepSeek), `mm` (MiniMax), `zhipu` 或 `or` (OpenRouter) 等子命令启动 Claude Code。
- **全局 vs 项目配置**: 使用 `use` 进行全局设置，使用 `use --project` 进行项目级模型覆盖。
- **会话管理**: 使用 `sess` (交互式 FZF App) 或 `resume` 管理并恢复之前的对话。
- **安全性**: 谨慎建议使用 `--dangerously-skip-permissions` 标志；仅在可信的沙盒环境中推荐。

## 补充场景
- **本地 LLM**: 通过 `x claude use other` 将 Claude Code 连接到本地 Ollama 服务。
- **Token 分析**: 运行 `x claude usage` 获取详细的成本和 Token 统计。
- **清洁 Git 历史**: 使用 `x claude attribution rm` 移除 Git 提交时自动生成的 Co-Author 签名。

## 实战示例

### 使用 DeepSeek 启动
```bash
# 使用 DeepSeek 模型启动 Claude Code
x claude ds
```

### 配置项目模型
```bash
# 将当前项目配置为使用智谱 (GLM) 模型提供商
x claude use --project glm
```

### 统计使用情况
```bash
# 分析过去 7 天的 Token 使用情况和成本
x claude usage
```

### 交互式会话管理
```bash
# 通过 FZF 浏览并管理本地会话
x claude sess
```

## 交付验证清单
- [ ] 确认用户是否想要使用特定的模型提供商（DeepSeek, MiniMax 等）。
- [ ] 验证配置应该是全局性的还是项目级的。
- [ ] 确保已为所选提供商配置了相关的 API Key。
