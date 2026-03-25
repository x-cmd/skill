---
name: luarocks
description: >
  Luarocks 的增强接口，Lua 模块的包管理器。
  核心场景：当用户需要安装、管理或搜索 Lua 依赖项时。
license: MIT
---

# luarocks - Lua 包管理

`luarocks` 模块提供了一个简化的命令行界面来管理 Lua 模块，并与 x-cmd 的包系统集成以便于安装。

## 激活时机
- 当安装或移除 Lua 软件包 (Rocks) 时。
- 当在 Luarocks 仓库中搜索模块时。

## 实战示例

### 安装模块
```bash
# 安装特定的 Lua 模块
x luarocks install luasocket
```

## 交付验证清单
- [ ] 确认模块名称。
- [ ] 验证 Lua 运行时是否已正确设置。
