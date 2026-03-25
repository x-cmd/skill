---
name: gddy
description: >
  从 CLI 管理 GoDaddy 域名和 DNS 记录，支持域名搜索和记录修改。
  核心场景：当用户需要审计其域名列表、检查可用性或编辑 DNS 记录时。
license: MIT
---

# gddy - GoDaddy 域名与 DNS 管理

`gddy` 模块为 GoDaddy API 提供了命令行界面，使用户能够从终端安全地管理其域名和 DNS 记录。

## 激活时机
- 当用户想要罗列其 GoDaddy 账户中的所有域名时。
- 当检查特定的域名是否可供注册时。
- 当为域名添加、移除或查看 DNS 记录（A, CNAME 等）时。
- 当管理 GoDaddy API 密钥和配置时。

## 核心原则与规范
- **API 凭据**: 提醒用户通过 `init` 或 `--cfg` 配置其 API key 和 secret。
- **域名搜索**: 如果子命令未被识别，该模块会自动将其视为域名可用性搜索。
- **破坏性编辑**: 移除 DNS 记录时请务必小心。

## 实战示例

### 列出域名
```bash
# 查看当前账户中的所有域名
x gddy domain ls
```

### 修改 DNS 记录
```bash
# 向域名添加一条新的 DNS 记录
x gddy domain record add --name dev --data "1.2.3.4" my-domain.com
```

### 域名可用性
```bash
# 检查特定域名是否可用
x gddy search example.com
```

## 交付验证清单
- [ ] 确保 GoDaddy API key 和 secret 已正确初始化。
- [ ] 确认目标域名和记录详情。
- [ ] 验证是打算在生产环境还是沙盒环境中执行。
