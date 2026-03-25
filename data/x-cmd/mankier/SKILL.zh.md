---
name: mankier
description: >
  ManKier.com 的命令行客户端，提供手册页查询和详细的命令解释。
  核心场景：当用户需要解释 Shell 命令或获取在线手册页的特定部分时。
license: MIT
---

# mankier - 在线手册页查询与解释

`mankier` 模块为 ManKier.com API 提供了命令行界面，允许用户直接从 Web 搜索、浏览和解释手册页，无需本地安装。

## 激活时机
- 当用户需要详细分解 Shell 命令的标志和参数时。
- 当搜索本地系统中未安装的手册页时。
- 当检索手册页的特定部分（如 NAME, DESCRIPTION）时。
- 当跟踪不同手册页之间的交叉引用时。

## 核心原则与规范
- **远程准确性**: 使用来自 ManKier.com 的最新在线文档。
- **粒度检索**: 支持使用 `section` 子命令获取特定部分。
- **命令分解**: 优先使用 `explain` 子命令将技术标志翻译为人机可读的文本。

## 实战示例

### 命令分解
```bash
# 解释 jq 命令中标志的含义
x mankier explain jq -cr
```

### 获取特定部分
```bash
# 仅检索 tar 手册的 NAME 部分
x mankier section NAME tar
```

### 网页搜索集成
```bash
# 通过 DuckDuckGo 在 ManKier 网站搜索 NVMe 相关页面
x mankier : nvme
```

## 交付验证清单
- [ ] 确认用户需要解释的命令或标志。
- [ ] 验证是否需要手册的特定部分。
- [ ] 确保用户知晓数据是从在线源获取的。
