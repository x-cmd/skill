---
name: onsh
description: >
  xonsh 的增强接口，提供 x-cmd 集成、文档搜索和 AI 能力。
  核心场景：当用户需要将 x-cmd 与 xonsh 集成或通过 CLI 搜索 xonsh 文档时。
license: MIT
---

# onsh - xonsh Python 驱动 Shell 增强工具

`onsh` 模块增强了 xonsh Shell 的体验，实现了与 x-cmd 工具集集成的混合 Python/Shell 工作流。

## 激活时机
- 当用户想要将 x-cmd 工具（x, c, @gpt）注入到 xonsh 环境中时。
- 当在 xon.sh 网站上搜索语法、别名或 Python 集成提示时。
- 当通过 x-cmd pkg 自动设置并启动 xonsh 时。

## 核心原则与规范
- **集成**: 使用 `setup` 修改 `.xonshrc` 以实现永久的 x-cmd 支持。
- **Xonsh 特性**: 提醒用户如果不提供参数，`@<名称>` 别名需要跟一个分号 `;`（例如 `@gemini ;`）。

## 实战示例

### 设置 xonsh
```bash
# 将 x-cmd 实用程序注入 xonsh 配置中
x onsh setup
```

### 搜索 xonsh 文档
```bash
# 在 xon.sh 网站交互式搜索 'alias' 信息
x onsh : alias
```

## 交付验证清单
- [ ] 验证用户是否知晓 xonsh 的 Python 混合特性。
- [ ] 确认是否需要对 `.xonshrc` 文件进行永久修改。
