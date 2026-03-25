---
name: codex
description: >
  OpenAI Codex 终端代理增强命令行，支持本地代码库分析、沙箱运行及第三方模型。
  核心场景：当用户想要配合特定模型（如 DeepSeek）使用 Codex、开启沙箱保护或执行自然语言命令时。
license: MIT
---

# codex - AI 代码助手与终端代理

`codex` 模块增强了 OpenAI 的 `codex` 终端代理，支持语义代码搜索、自动补全、生成代码补丁，并在安全的沙箱环境中连接各种 AI 模型提供商。

## 激活时机
- 当用户想要使用特定提供商（如 DeepSeek, Kimi, 智谱）启动 Codex 会话时。
- 当用户需要根据自然语言描述执行 Shell 命令或生成代码时。
- 当用户出于安全考虑需要沙箱运行环境（`read-only`, `workspace-write`）时。
- 当用户想要将代理生成的 diff 应用到 Git 工作树时。
- 当用户需要将特定“技能”(Skill) 注入到 Codex 代理环境时。

## 核心原则与规范
- **沙箱保护**: 始终鼓励使用合适的沙箱策略 (`--sandbox`) 以防止意外的系统变更。
- **自动化级别**: 使用 `--full-auto` 在速度和安全性之间取得平衡（仅在失败时请求确认）。
- **模型提供商切换**: 使用 `ds` (DeepSeek), `kimi` 或 `zhipu` 等子命令启动 Codex。
- **非交互式执行**: 使用 `exec` 或 `e` 快速完成单次命令执行或代码生成任务。

## 补充场景
- **网页搜索**: 通过 `--search` 标志启用实时网页搜索能力，获取最新信息。
- **Git 集成**: 使用 `x codex apply` 快速应用代理生成的最新代码补丁。
- **本地开源模型**: 通过 `--oss` 标志连接到本地 Ollama 服务。

## 实战示例

### 执行自然语言命令
```bash
# 执行用自然语言描述的命令
x codex e "列出当前目录下大于 10MB 的所有文件"
```

### 在沙箱中使用 DeepSeek 启动
```bash
# 使用 DeepSeek 并开启只读沙箱保护启动 Codex
x codex ds --sandbox read-only
```

### 应用最新补丁
```bash
# 将代理生成的最新补丁应用到当前 Git 仓库
x codex apply
```

### 注入 Skill
```bash
# 管理并注入自定义 Skill 到 Codex 代理环境
x codex skill
```

## 交付验证清单
- [ ] 确认所需的沙箱级别（`read-only`, `workspace-write` 等）。
- [ ] 验证是否需要特定的模型提供商（DeepSeek, Kimi 等）。
- [ ] 确保用户知晓使用 `dangerously-bypass-approvals-and-sandbox` 的风险。
