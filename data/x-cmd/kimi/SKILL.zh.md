---
name: kimi
description: >
  kimi-cli 增强模块，将 AI 编程能力集成到终端工作流中。
  核心场景：当用户想要启动 Kimi Code 代理进行编码辅助、升级工具或管理会话时。
license: MIT
---

# kimi - Kimi Code CLI 增强

`kimi` 模块增强了 `kimi-cli` 代理，为 AI 驱动的编码、会话管理和工作区集成提供无缝的终端体验。

## 激活时机
- 当用户想要启动 Kimi Code 交互式会话时。
- 当用户需要升级或管理 `kimi-cli` 工具时。
- 当将模型上下文协议 (MCP) 服务器与 Kimi 集成时。
- 当继续或分叉之前的 AI 编码会话时。

## 核心原则与规范
- **Yolo 模式**: 如果用户要求，使用 `-y` 或 `--yolo` 标志进行自动工具批准。
- **思考模式**: 使用 `--thinking` 或 `--no-thinking` 启用或禁用深度思考。
- **环境管理**: 使用 `--install` 或 `--upgrade` 确保工具是最新的。

## 补充场景
- **TUI 模式**: 使用 `x kimi term` 运行交互式终端 UI。
- **Web 界面**: 通过 `x kimi web` 启动 Kimi Web 界面。

## 实战示例

### 启动 Kimi Code
```bash
# 在当前目录下启动交互式 Kimi Code 会话
x kimi
```

### 自动批准命令
```bash
# 以自动批准所有操作的模式 (YOLO mode) 运行 Kimi
x kimi --yolo
```

### 继续上次会话
```bash
# 恢复当前工作区中最近的对话
x kimi --continue
```

## 交付验证清单
- [ ] 确保已安装 `kimi-cli`；如有必要，运行 `x kimi --install`。
- [ ] 确认用户是否想要启用自动批准 (YOLO)。
- [ ] 检查是否需要加载 MCP 配置。
