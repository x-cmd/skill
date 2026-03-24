# x-cmd 模块 Skill 编写计划列表

## AI & Agent (AI 与 智能体)
- [ ] `gram` - AI 终端伴侣
- [ ] `openclaw` - 开源 AI 助手
- [ ] `claude` - Anthropic Claude API 交互
- [ ] `codex` - 代码 AI 辅助
- [ ] `openspec` - 开放规范工具
- [ ] `speckit` - 规范工具包
- [ ] `openai` - OpenAI API 交互
- [ ] `gemini` - Google Gemini API 交互
- [ ] `deepseek` - DeepSeek API 交互
- [ ] `moonshot` - Moonshot (Kimi) API 交互
- [ ] `kimi` - Kimi AI 交互
- [ ] `doubao` - 字节跳动 豆包 AI 交互
- [ ] `zhipu` - 智谱 AI 交互
- [ ] `mistral` - Mistral AI 交互
- [ ] `grok` - xAI Grok API 交互
- [ ] `crush` - AI 驱动的数据处理
- [ ] `aider` - AI 结对编程助手
- [ ] `lms` - 本地 LLM 管理 (LM Studio)
- [ ] `llmf` - LLM 框架工具
- [ ] `ollama` - 本地运行 LLM 工具
- [ ] `whisper` - OpenAI Whisper 语音识别
- [ ] `jina` - Jina AI 搜索与处理
- [ ] `buse` - AI 业务逻辑封装
- [ ] `writer` - AI 写作辅助

## Standard Library (标准库扩展)
- [ ] `sudo` - 增强型 sudo
- [ ] `os` - 操作系统工具
- [ ] `is` - 类型/环境判断
- [ ] `tmp` - 临时文件管理
- [ ] `ccmd` - 命令组合与别名
- [ ] `tee` - 增强型 tee
- [ ] `str` - 字符串处理
- [ ] `rand` - 随机数与随机字符
- [ ] `assert` - 断言工具
- [ ] `humantime` - 人类可读时间转换
- [ ] `sleep` - 增强型 sleep
- [ ] `fsiter` - 文件系统迭代器

## OS Management (操作系统管理)
- [ ] `mac` - macOS 系统工具
- [ ] `termux` - Termux (Android) 工具
- [ ] `cd` - 增强型目录切换
- [ ] `top` - 进程监控
- [ ] `btop` - 交互式资源监控
- [ ] `htop` - 经典交互式进程监控
- [ ] `ps` - 增强型进程查看
- [ ] `id` - 用户标识信息
- [ ] `uname` - 系统信息
- [ ] `uptime` - 运行时间与负载
- [ ] `kill` - 增强型进程杀死工具
- [ ] `last` - 登录历史
- [ ] `pb` - 剪贴板管理 (Pasteboard)

## Hardware (硬件与资源)
- [ ] `cpu` - CPU 信息与状态
- [ ] `free` - 内存使用情况
- ✅ `df` - 磁盘空间查看
- [ ] `smart` - 磁盘健康 (S.M.A.R.T.)
- [ ] `ncdu` - 交互式磁盘占用分析

## File & Storage (文件与存储)
- ✅ `zuz` - 文件压缩与解压
- [ ] `ls` - 增强型列表查看
- [ ] `ll` - 详细列表查看
- [ ] `lsof` - 打开文件查看
- [ ] `path` - 路径处理
- [ ] `stat` - 文件状态信息
- [ ] `facl` - 文件访问控制列表

## Network (网络工具)
- [ ] `gg` - 简单的 Google 搜索或网络测试
- [ ] `ping` - 增强型 Ping
- [ ] `tping` - TCP Ping
- [ ] `hi` - 网络连通性测试
- [ ] `ip` - IP 地址与网络配置
- [ ] `arp` - ARP 表管理
- [ ] `dns` - DNS 查询与配置
- [ ] `nets` - 网络状态监控
- [ ] `host` - 主机名查询
- [ ] `hostname` - 系统主机名管理
- [ ] `route` - 路由表管理
- [ ] `proxy` - 代理配置与管理

## XaaS & Info (外部服务与信息)
- ✅ `mirror` - 开源镜像站工具
- [ ] `hub` - 中心化工具仓库
- [ ] `ws` - Web 服务工具
- [ ] `tldr` - 命令使用示例 (TL;DR)
- [ ] `man` - 增强型手册页
- [ ] `mankier` - 联机手册查询
- [ ] `cht` - cheat.sh 交互
- [ ] `hn` - Hacker News 浏览
- [ ] `se` - Stack Exchange / Stack Overflow
- [ ] `wkp` - 维基百科查询
- [ ] `ddgo` - DuckDuckGo 搜索
- ✅ `rfc` - IETF RFC 文档查询
- [ ] `gtb` - Project Gutenberg 电子书
- [ ] `coin` - 加密货币信息
- [ ] `coincap` - CoinCap 市场数据
- [ ] `ascii` - ASCII 字符参考与转换
- [ ] `wttr` - 天气查询 (wttr.in)
- [ ] `emoji` - Emoji 搜索与查找
- [ ] `hua` - 中文排版与处理

## Calendar & Data Tools (日历与数据工具)
- [ ] `ccal` - 中国农历
- [ ] `gcal` - Google 日历交互
- [ ] `cal` - 增强型日历
- [ ] `jq` - JSON 处理
- [ ] `yq` - YAML 处理
- [ ] `sed` - 流编辑器增强
- [ ] `sd` - 简单的查找替换 (Better sed)
- [ ] `grep` - 增强型文本搜索
- [ ] `rg` - ripgrep 集成
- [ ] `find` - 增强型文件查找

## Terminal & PKG Manager (终端与包管理)
- [ ] `theme` - 终端主题管理
- [ ] `starship` - Starship 提示符集成
- [ ] `ohmyposh` - Oh My Posh 集成
- [ ] `font` - 字体安装与管理
- [ ] `colr` - 终端颜色处理
- [ ] `pick` - 交互式选择工具
- [ ] `cowsay` - 牛牛说工具
- ✅ `install` - 软件安装工具
- [ ] `uninstall` - 软件卸载工具
- ✅ `env` - 环境管理 (envy)
- [ ] `cosmo` - Cosmopolitan Libc 工具
- [ ] `asdf` - 多版本运行环境管理
- [ ] `pixi` - Conda 兼容包管理
- [ ] `pkgx` - 现代包管理器
- [ ] `apt` - Debian/Ubuntu 包管理
- [ ] `pacman` - Arch Linux 包管理
- [ ] `aur` - Arch 用户仓库 (AUR)
- [ ] `paru` - AUR 助手
- [ ] `dnf` - Fedora/RHEL 包管理
- [ ] `yum` - 旧版 RHEL 包管理
- [ ] `brew` - Homebrew (macOS/Linux)
- [ ] `apk` - Alpine Linux 包管理
- [ ] `scoop` - Windows 包管理 (Scoop)
- [ ] `choco` - Windows 包管理 (Chocolatey)
- [ ] `winget` - Windows 原生包管理

## Message & Multimedia (消息与多媒体)
- [ ] `qywx` - 企业微信 API
- [ ] `discord` - Discord API
- [ ] `telegram` - Telegram Bot API
- [ ] `feishu` - 飞书 API
- [ ] `dingtalk` - 钉钉 API
- [ ] `ffmpeg` - 视频/音频处理
- [ ] `gm` - GraphicsMagick 图像处理
- [ ] `pandoc` - 文档转换工具

## Security, SSH & Cloud (安全与云原生)
- [ ] `osv` - 开源漏洞扫描 (Google OSV)
- [ ] `scorecard` - 开源安全评分
- [ ] `shodan` - Shodan 网络空间搜索
- [ ] `kev` - 已知被利用漏洞 (CISA KEV)
- [ ] `hash` - 各种哈希算法工具
- [ ] `hashdir` - 目录指纹计算
- [ ] `cowrie` - Cowrie 蜜罐工具相关
- [ ] `endlessh` - SSH 慢速连接工具相关
- [ ] `mosh` - 移动端远程 Shell
- [ ] `bwh` - BandwagonHost (搬瓦工) 管理
- [ ] `gddy` - GoDaddy 域名管理

## Git & Container (Git 与 容器)
- [ ] `gh` - GitHub CLI
- [ ] `gt` - Gitea CLI
- [ ] `gl` - GitLab CLI
- [ ] `tea` - Gitea CLI (alternative)
- [ ] `cb` - Codeberg CLI
- [ ] `fjo` - Forgejo CLI
- ✅ `git` - 增强型 Git
- [ ] `gitconfig` - Git 配置管理
- [ ] `githook` - Git 钩子管理
- [ ] `docker` - 容器管理
- [ ] `webtop` - 浏览器运行终端
- [ ] `sb` - SBOM (软件账单) 工具
- [ ] `lsio` - LinuxServer.io 相关工具

## Shell & Languages (Shell 与 编程语言)
- [ ] `nu` - Nushell 集成
- [ ] `elv` - Elvish Shell 集成
- [ ] `fish` - Fish Shell 集成
- [ ] `onsh` - 只有一行代码的 Shell 相关
- [ ] `tcsh` - Tcsh 集成
- [ ] `pwsh` - PowerShell 集成
- [ ] `zig` - Zig 语言工具
- [ ] `lua` - Lua 语言环境
- [ ] `luarocks` - Lua 包管理
- [ ] `python` - Python 环境与包管理
- [ ] `pip` - Python 包安装
- [ ] `node` - Node.js 环境
- [ ] `npm` - Node 包管理
- [ ] `npx` - Node 执行工具
- [ ] `deno` - Deno 运行环境
- [ ] `bun` - Bun 运行环境
- [ ] `perl` - Perl 环境
- [ ] `raku` - Raku 环境
- [ ] `java` - Java (JDK) 环境
- [ ] `groovy` - Groovy 语言工具
