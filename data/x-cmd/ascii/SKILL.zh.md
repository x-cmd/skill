---
name: ascii
description: >
  处理与 ASCII 相关的任务，包括码表、艺术字、图片转换和终端渲染（Mermaid）。
  核心场景：当用户需要查看 ASCII 码、生成 ASCII 艺术或在终端渲染 Mermaid 图表时。
license: MIT
---

# ascii - ASCII 实用工具与终端艺术

`ascii` 模块提供了一套用于处理 ASCII 文本和视觉资产的工具。它涵盖了从标准 ASCII 码表到复杂的终端渲染（如世界地图和 Mermaid 图表）的所有内容。

## 激活时机
- 当用户想要查找 ASCII 字符代码（十进制、十六进制）时。
- 当生成 ASCII 艺术字 (cfont) 或将图片转换为 ASCII 艺术画时。
- 当在终端渲染 Mermaid 图表或绘制 ASCII 折线图时。
- 当查看基于 ASCII 的世界地图或运行 ASCII 动画（烟花）时。

## 核心原则与规范
- **动态执行**: 许多子命令（如 `cfont` 和 `mermaid`）通过 deno 或 npm 按需下载所需的工具。
- **视觉创造力**: 支持使用颜色和对齐方式设置艺术字的样式。
- **管道支持**: 可以根据管道传输的数字数据绘制图表（例如 `seq 1 10 | x ascii graph`）。

## 实战示例

### ASCII 码表
```bash
# 查看标准 ASCII 码表
x ascii table
```

### ASCII 艺术字
```bash
# 将文本转换为彩色的 ASCII 艺术字
x ascii cfont x-cmd -g red,blue
```

### Mermaid 渲染
```bash
# 在终端中直接渲染 Mermaid 图表
x ascii mermaid
```

### 数据绘图
```bash
# 从序列创建 ASCII 折线图
seq 1 10 | x ascii graph
```

## 交付验证清单
- [ ] 确认用户需要的是代码参考还是视觉转换。
- [ ] 验证 ASCII 艺术是否请求了特定的颜色或样式。
- [ ] 确保终端宽度足以进行大型 ASCII 渲染。
