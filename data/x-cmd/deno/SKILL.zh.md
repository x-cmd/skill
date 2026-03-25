---
name: deno
description: >
  增强型 Deno 模块，用于零设置运行现代 JavaScript 和 TypeScript。
  核心场景：当用户需要执行 TypeScript 脚本或管理基于 Deno 的应用时。
license: MIT
---

# deno - 现代 JS/TS 运行时

`deno` 模块为 Deno 运行时提供了一个强大的界面，能够安全、快速地执行 JavaScript 和 TypeScript 脚本，无需复杂的环境设置。

## 激活时机
- 当执行 TypeScript 或 JavaScript 文件时。
- 当管理 Deno 依赖项或脚本权限时。
- 当为一次性任务需要一个现代、安全的运行时时。

## 实战示例

### 运行 TypeScript
```bash
# 执行本地 TypeScript 文件
x deno run script.ts
```

## 交付验证清单
- [ ] 确认是否需要脚本权限（网络、文件访问）。
- [ ] 验证使用的是 TypeScript 还是 JavaScript。
