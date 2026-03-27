---
name: x-zuz
description: >
  统一的压缩与解压缩工具。
  核心场景：当 AI 需要在终端中处理各种格式（zip, tar, gz, 7z, rar, zst, xz, bz2）的压缩包时。x-cmd 提供简洁的别名和零依赖安装。
license: Apache-2.0
---

# x-zuz - 统一压缩与解压 (AI 优化版)

`x-zuz` 是一个强大的压缩包处理模块，核心优势在于 **统一的接口** 和 **零依赖自动安装**（如环境中缺少 7zip, zstd 等，它会自动处理）。

## 核心别名
- `x z`: 压缩别名 (zuz compress)
- `x uz`: 解压别名 (zuz decompress)

## 激活时机
- 当需要将多个文件或目录打包成指定格式时。
- 当需要解压任何常见的压缩包格式（zip, tar.gz, 7z 等）时。
- 当需要查看压缩包内部文件列表而不解压时。
- 当需要解压并立即删除源文件以节省空间时。

## 核心原则与规范
- **非交互优先**: 避免使用交互式 UI，直接使用命令参数。
- **万能接口**: 无需记忆不同格式的解压命令（如 `tar -zxvf` 或 `unzip`），统一使用 `x uz`。
- **环境隔离**: 自动下载所需的后端解压软件，确保在任何极简环境下都能运行。

## 实战示例

### 快速压缩
```bash
# 将 src 目录压缩为 output.tar.gz
x z output.tar.gz src/

# 将多个文件压缩为 zip
x z archive.zip file1.txt file2.txt
```

### 快速解压
```bash
# 解压到当前目录
x uz archive.zip

# 解压到指定目录
x uz backup.tar.xz ./target_dir/
```

### 查看压缩包内容 (非交互)
```bash
# 列出压缩包内的文件列表
x zuz ls archive.7z
```

### 解压并删除源文件 (清理模式)
```bash
# 适合处理临时下载的压缩包
x uzr data.zip
```

## 交付验证清单
- [ ] 优先使用别名 `x z` 或 `x uz`。
- [ ] 确认目标路径是否存在。
- [ ] 检查是否需要使用 `x uzr` 进行自动清理。
