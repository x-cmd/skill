---
name: zig
description: >
  增强型 Zig 语言模块，用于包管理、项目构建及 ZON 格式转换。
  核心场景：当用户需要管理 Zig 项目、将 ZON 转换为 JSON/YAML 或使用 Zig 初始化 C 编译时。
license: MIT
---

# zig - Zig 语言增强工具

`zig` 模块扩展了 Zig 工具链的能力，提供了简化的包管理、针对 `build.zig.zon` 的强大格式转换以及集成的 C 编译实用程序。

## 激活时机
- 当管理 Zig 包并搜索常用库时 (`pm`)。
- 当将 `build.zig.zon` 文件转换为 JSON 或 YAML 以便分析时。
- 当使用 Zig 作为标准编译器的直接替代品初始化 C 编译时 (`initcc`)。
- 当执行标准的 Zig 任务（如构建、格式化或测试项目）时。

## 核心原则与规范
- **格式转换**: 使用 `zon` 子命令将 Zig 的内部格式转换为机器可读的 JSON/YAML。
- **C 集成**: 利用 `cc`、`c++` 和 `ar` 将 Zig 内置的工具链用于 C 项目。
- **包管理**: 使用 `pm la` 发现常用的 Zig 包。

## 实战示例

### 将 ZON 转换为 YAML
```bash
# 将 build.zig.zon 内容以 YAML 格式输出
cat build.zig.zon | x zig zon toyml
```

### 发现包
```bash
# 列出 x-cmd 收集的所有可用 Zig 包
x zig pm la
```

### 初始化 C 编译
```bash
# 为 C 项目编译设置 Zig
x zig initcc
```

## 交付验证清单
- [ ] 确认用户需要的是 Zig 原生任务还是 C 集成实用程序。
- [ ] 验证用于构建或格式化的目标文件路径。
- [ ] 检查 ZON 数据转换时首选的是 JSON 还是 YAML。
