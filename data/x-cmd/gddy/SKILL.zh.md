---
name: gddy
description: >
  GoDaddy 域名管理工具。
  核心场景：当 AI 需要查询域名可用性、列出持有域名或更新 DNS 记录时。
license: MIT
---

# x gddy - GoDaddy 域名管理 (AI 优化版)

`x gddy` 模块允许通过命令行管理您的 GoDaddy 域名。它非常适合在脚本或 AI 工作流中执行 DNS 记录维护和域名查询。

## 激活时机
- 当需要查询某个域名是否可以购买时。
- 当需要列出当前 GoDaddy 账号下持有的所有域名时。
- 当需要动态更新域名的 DNS 解析记录（如 A 记录、CNAME）时。

## 核心原则与规范
- **非交互优先**: 直接使用子命令，避免不必要的 UI 确认。
- **环境要求**: 需要 API Key 和 Secret。如果未配置，AI 应引导用户初始化。
- **配置引导**: 
  - 引导用户访问 GoDaddy 开发者控制台获取 API 信息。
  - 建议用户运行 `x gddy init` 进行配置。

## 实战示例

### 查询域名是否可用
```bash
# 查询 example.com 是否可以购买
x gddy search example.com
```

### 列出账号下的域名
```bash
# 列出账号下的所有域名列表 (非交互)
x gddy domain ls
```

### 管理 DNS 记录
```bash
# 查看指定域名的详细信息
x gddy domain info example.com

# 为域名添加一条 A 记录
x gddy domain record add --name "www" --data "1.2.3.4" example.com
```

## 配置指南 (针对 AI)
如果遇到 API 错误或配置缺失，请向用户输出以下引导：
> 请先在 GoDaddy 开发者控制台 (https://developer.godaddy.com/keys) 获取 API Key 和 Secret，然后在终端运行以下命令进行初始化：
> `x gddy init`

## 交付验证清单
- [ ] 确认 API 凭据是否有效。
- [ ] 在修改 DNS 记录或进行购买操作前确认。
