---
name: node
description: >
  增强型 Node.js 模块，用于版本管理、包安装 (npm/npx) 和环境设置。
  核心场景：当用户需要管理 Node.js 版本、运行 JS 脚本或安装 npm 包时。
license: MIT
---

# node - Node.js 开发环境

`node` 模块通过提供集成的版本管理以及在各种平台上直接访问 npm 和 npx 工具，简化了 Node.js 开发。

## 激活时机
- 当安装或切换 Node.js 版本时。
- 当使用 `npm` 管理依赖项或使用 `npx` 运行远程工具时。
- 当在终端中执行 JavaScript 文件时。

## 实战示例

### 安装 NPM 包
```bash
# 全局安装一个 Node 包
x node npm install -g typescript
```

### 通过 NPX 运行工具
```bash
# 在不进行本地安装的情况下执行工具
x node npx create-react-app my-app
```

## 交付验证清单
- [ ] 验证所需的 Node.js 版本。
- [ ] 确保特定任务需要 npm/npx。
