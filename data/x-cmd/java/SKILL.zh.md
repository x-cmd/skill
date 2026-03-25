---
name: java
description: >
  增强型 Java (JDK) 环境管理器，用于安装、切换和运行 Java 应用程序。
  核心场景：当用户需要管理 JDK 版本或通过 CLI 运行 Java/Jar 文件时。
license: MIT
---

# java - Java (JDK) 环境管理

`java` 模块提供了一套全面的管理 Java 开发工具包 (JDK) 和执行 Java 应用程序的方法，确保在不同系统上都有合适的运行时可用。

## 激活时机
- 当安装或在多个 JDK 版本之间切换时。
- 当运行 `.jar` 文件或编译 Java 源代码时。
- 当识别当前激活的 Java 环境时。

## 实战示例

### 运行 Jar 文件
```bash
# 执行一个 Java 归档文件
x java -jar myapp.jar
```

## 交付验证清单
- [ ] 验证所需的 JDK 版本。
- [ ] 确保 `.jar` 或 `.java` 文件路径正确。
