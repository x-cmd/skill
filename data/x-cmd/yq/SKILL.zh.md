---
name: yq
description: >
  便携式命令行 YAML、JSON 和 XML 处理器，具有用于探索数据的交互式 REPL。
  核心场景：当用户需要查询、提取或交互式地浏览 YAML 配置或结构化数据时。
license: MIT
---

# yq - YAML, JSON 和 XML 处理器

`yq` 模块是一个处理结构化数据格式（主要是 YAML）的通用工具。x-cmd 版本增加了由 FZF 驱动的交互式 REPL，使浏览复杂的配置文件变得直观。

## 激活时机
- 当用户需要查询或修改 YAML 配置文件时。
- 当在 YAML、JSON 和其他格式之间进行转换时。
- 当通过实时反馈交互式地探索嵌套的 YAML 结构时。
- 当从深度嵌套的键中提取特定值时。

## 核心原则与规范
- **交互式探索**: 使用 `repl` (或 `r`) 子命令通过可搜索的 TUI 浏览 YAML 数据。
- **多格式支持**: 支持 YAML、JSON、XML 和属性文件作为输出或输入格式。
- **原地编辑**: 支持通过 `-i` 标志直接修改文件。

## 实战示例

### 交互式探索配置
```bash
# 使用 FZF 浏览复杂的配置文件
x yq r config.yml
```

### 提取值
```bash
# 从 YAML 文件中获取特定的嵌套值
x yq '.database.host' config.yml
```

### 转换格式
```bash
# 将 YAML 文件输出为 JSON
x yq -o json config.yml
```

## 交付验证清单
- [ ] 确认目标文件是 YAML、JSON 还是 XML。
- [ ] 验证需要的是交互式视图还是特定的提取。
- [ ] 检查文件是否应原地修改 (`-i`)。
