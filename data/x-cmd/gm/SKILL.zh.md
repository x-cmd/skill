---
name: gm
description: >
  GraphicsMagick 的增强型接口，提供强大的图像处理和转换工具。
  核心场景：当用户需要转换图像格式、调整照片大小或执行批量图像编辑时。
license: MIT
---

# gm - GraphicsMagick 图像处理器

`gm` 模块为 GraphicsMagick 提供了一个增强型界面，允许用户高效地处理图像。如果本地未找到该工具，它会自动通过 pixi 处理安装。

## 激活时机
- 当在不同格式之间转换图像时（例如 JPG 转 PNG）。
- 当调整图像大小或执行批量转换时。
- 当比较两张图像或创建图像蒙太奇时。
- 当需要图像识别和元数据描述时。

## 核心原则与规范
- **工具可靠性**: 使用 `pixi` 确保 GraphicsMagick 在不同环境中均可用。
- **标准命令**: 支持所有经典的 `gm` 子命令，如 `convert`, `mogrify` 和 `identify`。

## 实战示例

### 转换图像
```bash
# 将 JPG 图像转换为 PNG 格式
x gm convert test.jpg test.png
```

### 调整图像大小
```bash
# 将图像大小调整为 300 像素宽
x gm convert -resize 300 test.jpg output.jpg
```

### 列出格式
```bash
# 查看转换所支持的所有图像格式
x gm convert -list formats
```

## 交付验证清单
- [ ] 确认目标操作（转换、调整大小等）。
- [ ] 验证输入文件路径和格式。
