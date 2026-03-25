---
name: elv
description: >
  Elvish Shell 的增强接口，支持 x-cmd 集成、官方文档搜索和 AI 命令生成。
  核心场景：当用户需要将 x-cmd 与 Elvish 集成或通过终端搜索 Elvish 文档时。
license: MIT
---

# elv - Elvish Shell 增强工具

`elv` 模块通过提供与 x-cmd 的无缝集成以及快速查找 Elvish 文档，简化了 Elvish Shell 的使用。

## 激活时机
- 当用户想要在 Elvish Shell 中使用 x-cmd 工具（x, c, @gpt）时。
- 当在 elv.sh 网站上搜索语法、模块或别名定义时。
- 当通过 x-cmd 包管理器零设置启动 Elvish 时。

## 核心原则与规范
- **集成**: 使用 `--setup` 将 x-cmd 添加到 Elvish 环境中。
- **动态获取**: 如果未预装 Elvish，会自动获取。

## 实战示例

### 注入 x-cmd
```bash
# 在 Elvish Shell 环境中设置 x-cmd 工具
x elv --setup
```

### 文档查找
```bash
# 在 Elvish 网站上搜索 'alias' 示例
x elv : alias
```

## 交付验证清单
- [ ] 确保用户熟悉 Elvish 的结构化数据处理方式。
- [ ] 确认是否需要对 Elvish 配置文件进行永久修改。
