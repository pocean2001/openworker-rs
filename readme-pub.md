# OpenWorker-rs

> 吴恩达老师（Andrew Ng）[OpenWorker](https://github.com/andrewyng/openworker) 核心 Agent 引擎的 **Rust 重写版本**。

OpenWorker-rs 用 Rust 技术栈完整重写了 OpenWorker 的后端引擎，目标是把一个"本地优先、隐私友好"的 AI 同事（coworker）跑在单一可执行文件里——无需浏览器、无需本地服务、无需 WebView 运行时。

## 它是什么

一个命令行 + 原生桌面的 AI 工作伙伴：你给它任务，它自己规划、调用工具、读写文件、运行命令、检索网络、调用 MCP 服务，最后把"做完的活"交还给你，而不只是陪你聊天。

- **本地优先 / 隐私友好**：密钥与配置只存在你本机，不上传第三方。
- **单一二进制**：一个 `openworker` 可执行文件，同时提供 CLI 和原生 GUI。
- **Agent 循环**：模型 ⇄ 工具自主迭代，支持文件读写、命令执行、网页检索、MCP 工具、可自写技能（skills）等。
- **定时自动化**：用标准 cron 配置周期性任务（如每日晨报）。
- **原生 GUI 客户端**：开箱即用的桌面窗口，左侧栏即可配置 API 并一键测试连接。

## 快速开始

环境要求：Rust 工具链 + Windows GNU 工具链（MinGW）。

```bash
cargo build            # 编译（Windows 下建议用仓库内的 build.sh，自动注入 MinGW 路径）
cargo run -- run       # 进入交互式对话
cargo run -- gui       # 启动原生桌面客户端
cargo run -- automation list   # 查看已配置的定时任务
```

## 配置与密钥（重要）

仓库**只提交示例配置** `openworker.toml`，其中不含任何密钥。

你的真实密钥放在 **`openworker.local.toml`**，该文件已被 `.gitignore` 忽略，**不会进入版本库**。你也可以用环境变量注入密钥：

- `DEEPSEEK_API_KEY`（默认协议为 DeepSeek）
- `OPENAI_API_KEY`

GUI 客户端的左侧栏已内置 **API 配置**面板：默认采用 DeepSeek 协议，可填写 Key / Base URL / 模型，并点击"测试连接"一键验证可用性。

## 致谢

本项目是 Andrew Ng 与 OpenWorker 项目的 Rust 重写练习与实现，向其原创设计与开源精神致以敬意。
