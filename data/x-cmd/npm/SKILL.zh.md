---
name: npm
description: >
  npm 的增强接口，Node.js 软件包管理器。
  核心场景：当用户需要管理 JavaScript 依赖项或运行 npm 脚本时。
license: MIT
---

# npm - Node.js 包管理

`npm` 模块提供了对 Node 软件包管理器的集成访问，简化了 Node.js 项目的依赖项安装和脚本执行。

## 激活时机
- 当安装或管理 Node.js 软件包时。
- 当运行 `package.json` 中定义的脚本时。

## 实战示例

### 安装全局包
```bash
# 通过 npm 全局安装一个包
x npm install -g nodemon
```

## 交付验证清单
- [ ] 确认包名。
- [ ] 验证目标项目目录。
