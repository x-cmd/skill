---
name: jq
description: >
  轻量级且灵活的 JSON 处理器。
  核心场景：当 AI 需要在终端中过滤、转换、格式化或提取 JSON 数据时。x-cmd 版 jq 提供零依赖自动安装。
license: MIT
---

# x jq - JSON 处理器 (AI 优化版)

`x jq` 是基于 jq 的增强模块，核心优势在于 **零依赖自动安装** 和 **针对脚本化任务的优化**。它能确保在任何环境下都能立即处理 JSON。

## 激活时机
- 当需要从复杂的 JSON 响应中提取特定字段时。
- 当需要过滤、重构或转换 JSON 结构时。
- 当需要将非格式化的 JSON 字符串美化以供后续分析时。
- 当需要将处理结果以原始字符串格式（`-r`）输出给其他命令时。

## 核心原则与规范
- **非交互优先**: AI 应当避免使用交互式 `repl` 模式，应直接使用 jq 表达式。
- **管道集成**: 建议配合管道使用，如 `cat data.json | x jq '.field'`。
- **格式控制**: 
  - 使用 `-r` (raw-output) 输出不带引号的原始值（适合获取单个字符串值）。
  - 使用 `-c` (compact-output) 输出紧凑的一行 JSON，节省上下文 Token。
- **环境隔离**: `x jq` 会在必要时自动下载并运行 jq，不会污染系统环境。

## 实战示例

### 提取并输出原始字符串
```bash
# 获取 'version' 字段的值（不带引号，适合后续脚本使用）
x jq -r '.version' package.json
```

### 过滤数组并紧凑输出
```bash
# 过滤并以一行紧凑 JSON 输出，节省 Token
x jq -c '.items[] | select(.status == "active")' data.json
```

### 构造新的 JSON 对象
```bash
# 构造一个包含 status 和 timestamp 的新对象
x jq -n --arg ts "$(date)" '{"status": "ok", "timestamp": $ts}'
```

### 处理多个文件
```bash
# 合并并处理多个 JSON 文件
x jq -s '.[0] * .[1]' config1.json config2.json
```

## 交付验证清单
- [ ] 确认使用的是非交互式命令（不带 `r` 或 `repl`）。
- [ ] 考虑是否需要使用 `-r` 来获取纯文本值。
- [ ] 考虑是否需要 `-c` 来减小输出体积。
