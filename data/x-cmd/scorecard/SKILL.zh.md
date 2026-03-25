---
name: scorecard
description: >
  用于评估开源项目风险和最佳实践遵循情况的自动化安全工具。
  核心场景：当用户需要评估 GitHub 仓库或软件包的安全健康状况时。
license: MIT
---

# scorecard - OpenSSF 安全记分卡

`scorecard` 模块根据安全最佳实践评估开源项目，提供评分以及关于潜在风险（如二进制工件、未审计的代码或危险的工作流）的详细报告。

## 激活时机
- 当用户想要评估开源仓库的安全级别时。
- 当对新的依赖项（npm, PyPI 等）执行尽职调查时。
- 当审计本地仓库以寻求安全改进时。

## 核心原则与规范
- **最佳实践**: 专注于识别缺乏 CI 测试、缺少分支保护或未锁定依赖项等风险。
- **详细报告**: 使用 `--show-details` 来了解特定检查通过或失败的原因。

## 实战示例

### 仓库审计
```bash
# 显示 GitHub 仓库的 OpenSSF 安全记分卡
x scorecard info github.com/ossf/scorecard
```

### 打开网页报告
```bash
# 在浏览器中打开完整的 OpenSSF 记分卡报告
x scorecard open github.com/owner/repo
```

## 交付验证清单
- [ ] 确认目标仓库 URL 或软件包名称。
- [ ] 验证用户需要的是摘要还是详细的检查分解。
