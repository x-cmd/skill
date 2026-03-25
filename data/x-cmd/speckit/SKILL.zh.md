---
name: speckit
description: >
  Spec Kit 增强模块，帮助用户设置并使用 Specify CLI 进行规范驱动开发。
  核心场景：当用户想要初始化一个 Specify 项目，以规范化 “需求→计划→任务→实现” 流程时。
license: MIT
---

# speckit - Spec Kit & Specify CLI 增强

`speckit` 模块促进了 Spec Kit 和 Specify CLI 的使用，它们引导开发者和 AI 工具通过标准化的、规范驱动开发流程，确保高质量、可预测的产出。

## 激活时机
- 当用户想要初始化一个新的 Specify 项目以使开发过程正规化时。
- 当用户需要为 Specify 项目设置特定的 AI 助手（如 Claude）时。
- 当用户想要检查 Spec Kit 所需的所有工具是否已安装时。

## 核心原则与规范
- **正规化工作流**: 强调从需求到实现的流程，以消除不确定性。
- **AI 特定初始化**: 支持使用特定的 AI 助手（如 `claude` 或 `codex`）初始化项目。

## 补充场景
- **工具验证**: 使用 `check` 确保环境已准备好进行规范驱动开发。

## 实战示例

### 初始化项目 (交互式)
```bash
# 通过交互式 AI 选择设置一个 Specify 项目
x speckit init
```

### 指定 AI 进行初始化
```bash
# 在指定目录中设置一个以 Claude 为助手的项目
x speckit init my-project --ai claude
```

## 交付验证清单
- [ ] 确认在初始化过程中是否需要特定的 AI 助手。
- [ ] 运行 `x speckit check` 确保所有必需的工具均可用。
