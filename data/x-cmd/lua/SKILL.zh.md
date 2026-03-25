---
name: lua
description: >
  增强型 Lua 开发模块，支持项目初始化、通过 Luarocks 安装模块以及静态编译。
  核心场景：当用户需要设置 Lua 项目、管理库或将 Lua 脚本编译为静态二进制文件时。
license: MIT
---

# lua - 增强型 Lua 开发工具

`lua` 模块为 Lua 开发者提供了一套全面的工具，简化了项目设置、依赖项管理和脚本执行。

## 激活时机
- 当初始化一个新的 Lua 项目结构时 (`init`)。
- 当使用集成的 Luarocks 安装 Lua 模块和库时 (`install`)。
- 当格式化、检查或对 Lua 源代码进行 lint 时。
- 当将 Lua 脚本编译为独立的静态二进制文件时。

## 核心原则与规范
- **Luarocks 集成**: 使用 `install` (或 `i`) 子命令轻松管理依赖项。
- **静态编译**: 利用 `static` 子命令创建零依赖的 Lua 可执行文件。
- **代码质量**: 支持使用 `check` 和 `format` 维护 Lua 代码标准。

## 实战示例

### 安装库
```bash
# 使用集成的 Luarocks 安装 'lua-cjson' 模块
x lua i lua-cjson
```

### 静态构建
```bash
# 将 Lua 脚本编译为单个静态二进制文件
x lua static main.lua
```

### 初始化项目
```bash
# 设置一个新的 Lua 项目环境
x lua init
```

## 交付验证清单
- [ ] 确认用户需要管理依赖项还是执行脚本。
- [ ] 验证静态编译的目标平台。
