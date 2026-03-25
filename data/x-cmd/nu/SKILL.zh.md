---
name: nu
description: >
  Nushell 的增强接口，支持 x-cmd 集成、文档搜索和 AI 驱动的命令生成。
  核心场景：当用户需要在 Nushell 中设置 x-cmd 或通过 CLI 搜索 Nushell 文档时。
license: MIT
---

# nu - Nushell 增强与集成

`nu` 模块通过提供与 x-cmd 生态系统的轻松集成，以及快速访问 Nushell 官方文档和 AI 生成的命令，提升了 Nushell 的体验。

## 激活时机
- 当用户想要将 x-cmd 工具（x, c, @gpt）注入到其 Nushell 环境中时。
- 当在 nushell.sh 网站上搜索特定的 Nushell 语法或别名信息时。
- 当通过 x-cmd pkg 自动安装并启动 Nushell 时。

## 核心原则与规范
- **集成**: 使用 `--setup` 自动修改 `env.nu` 以支持 x-cmd。
- **按需安装**: 如果系统中缺少 Nushell，会自动处理其安装。

## 实战示例

### 设置 x-cmd
```bash
# 将 x-cmd 实用程序注入 Nushell 配置中
x nu --setup
```

### 搜索文档
```bash
# 在 nushell.sh 网站交互式搜索 'alias' 信息
x nu : alias
```

## 交付验证清单
- [ ] 确认用户是否将 Nushell 作为其主要 Shell。
- [ ] 验证 x-cmd 应该是永久注入还是仅进行测试。
