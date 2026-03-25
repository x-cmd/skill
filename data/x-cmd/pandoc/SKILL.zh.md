---
name: pandoc
description: >
  万能文档转换器，支持在 Markdown, HTML, PDF 等格式之间进行转换。
  核心场景：当用户需要转换文档格式或将网页内容获取为 Markdown 时。
license: MIT
---

# pandoc - 万能文档转换器

`pandoc` 模块为强大的 Pandoc 文档转换器提供了一个界面。它简化了安装过程，并能轻松地在多种文本和文档格式之间进行转换。

## 激活时机
- 当将 Markdown 文件转换为 HTML、PDF 或 Word 文档时。
- 当获取网页内容并将其转换为干净的 Markdown 时。
- 当列出支持的文档格式或扩展名时。
- 当需要一个零设置的文档转换工具时。

## 核心原则与规范
- **零设置**: 如果二进制文件缺失，会自动下载并管理 pandoc。
- **多功能性**: 强调其处理“任何格式到任何格式”转换的能力。
- **独立支持**: 鼓励使用 `-s` 标志以获取带有页眉/页脚的完整文档。

## 实战示例

### Markdown 转 HTML
```bash
# 将 Markdown 文件转换为独立的 HTML 页面
x pandoc -s input.md -o output.html
```

### 网页转 Markdown
```bash
# 将在线网页转换为 Markdown 文档
x pandoc -s -r html https://example.com -o webpage.md
```

### 列出格式
```bash
# 查看所有支持的输入文档格式
x pandoc --list-input-formats
```

## 交付验证清单
- [ ] 确认源文档和目标文档的格式。
- [ ] 验证是需要独立文档 (`-s`) 还是代码片段。
- [ ] 确保输入 URL 或文件路径正确。
