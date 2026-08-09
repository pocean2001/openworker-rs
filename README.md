# openworker-rs

A **Rust rewrite of the OpenWorker core agent engine** (Andrew Ng 的 [andrewyng/openworker](https://github.com/andrewyng/openworker))。

OpenWorker 本身是一个“本地优先的 AI 同事”：它不是聊天机器人，而是把你要的结果拆成步骤、跨本地文件和已连接的应用执行，最终**交付成品**（文档、报告、改好的日历、整理过的收件箱）。它的后端（`coworker/`）是 Python 写的。**本项目用 Rust 重写了这套后端核心**——模型接入、带工具调用的 Agent 循环、MCP 客户端、记忆、自动化——并用一个 CLI 把它跑起来。

> 范围：本次重写聚焦 `coworker/` 后端核心（引擎 + 模型层 + 工具 + MCP + 记忆 + 自动化）与 CLI。未包含 React/Tauri 桌面壳（`surfaces/gui/`，那是前端）与语音组件（`stt/`，本来就是 Rust）。

## 与上游的对应关系

| 上游（Python） | 本仓库（Rust） | 说明 |
|---|---|---|
| `coworker/providers/base.py` + `openai_provider.py` | `src/provider.rs` | `ProviderClient` 抽象；`OpenAICompatibleProvider` 走 OpenAI Chat Completions 线路，Ollama / DeepSeek / Qwen / GLM / Mistral 等兼容端点通用；含 SSE 流式 + 文本工具调用“抢救” |
| `coworker/engine.py` (`TurnEngine`) | `src/engine.rs` | 模型⇄工具循环；实时 delta 事件；审批门控；低风险工具并发、写/Shell 严格串行 |
| `coworker/tools.py` | `src/tools.rs` | `Tool` trait + `ToolRegistry`；内置 `read_file / write_file / list_dir / run_command / web_fetch / ask_user` |
| `coworker/mcp/client.py` + `tools.py` | `src/mcp.rs` | stdio JSON-RPC 2.0 MCP 客户端；把任意 MCP 服务器的工具桥接进 `ToolRegistry`（命名 `mcp__<server>__<tool>`） |
| `coworker/memory/sqlite_store.py` | `src/memory.rs` | 会话记忆。上游用 SQLite，**本仓库改用零依赖的 JSONL 存储**（每条消息一行），避免 C 编译器依赖，也更易检视 |
| `coworker/automation/*` | `src/automation.rs` | cron 驱动的自动化调度器 |
| `coworker/permissions.py` | `src/permissions.rs` | 审批策略：`Interactive / Auto / Plan / Discuss` 模式 |
| `coworker/server/run.py` CLI | `src/main.rs` | `run / mcp / automation` 子命令 |

## 构建与运行

需要 Rust 工具链（[rustup](https://rustup.rs)）。**推荐用仓库自带的包装脚本**，它会自动挂好
工具链和 MinGW（原因见下一节）：

```bash
cd openworker-rs
./build.sh             # debug
./build.sh --release   # release
./build.sh check       # 只做类型检查
./build.sh test        # 跑单元测试
```

如果你的环境里 `gcc` 已经在 PATH 上，直接 `cargo build --release` 也可以。

### Windows (`x86_64-pc-windows-gnu`) 工具链要求

实测结论（本机无 MSVC / Windows SDK）：

| 需求 | 是否必须 | 原因 |
|---|---|---|
| 链接纯 Rust crate | ❌ 不需要额外装 | Rust 自带 `rust-lld` + self-contained mingw 运行库，hello-world 可直接链接 |
| `ring`（rustls 的加密后端） | ✅ **需要 gcc** | 含 C 源码，build script 通过 `cc` crate 调用 |
| `windows-sys` 的 raw-dylib 导入库 | ✅ **需要 dlltool** | rustc 生成 import lib 时外部调用 `dlltool.exe` |

所以 Windows 上仍需一套 GNU binutils/gcc。本机用的是 **MSYS2 MinGW-w64**（`D:\msys64\mingw64`，
gcc 16.1.0 / binutils 2.46 / `x86_64-w64-mingw32`）。

路径已固化在 [`.cargo/config.toml`](.cargo/config.toml)：

```toml
[target.x86_64-pc-windows-gnu]
linker = "D:\\msys64\\mingw64\\bin\\gcc.exe"
ar     = "D:\\msys64\\mingw64\\bin\\ar.exe"
rustflags = ["-Cdlltool=D:\\msys64\\mingw64\\bin\\dlltool.exe"]

[env]
CC  = "D:\\msys64\\mingw64\\bin\\gcc.exe"
CXX = "D:\\msys64\\mingw64\\bin\\g++.exe"
AR  = "D:\\msys64\\mingw64\\bin\\ar.exe"
```

#### ⚠️ 但绝对路径还不够——PATH 仍然必须设

这是本次踩得最深的坑，记录一下：

MSYS2 的 `gcc.exe` 会去 `lib\gcc\x86_64-w64-mingw32\<ver>\` 启动 `cc1.exe`，而 `cc1.exe`
需要的 `libisl / libmpc / libmpfr / libgmp / zlib` 这些 DLL 却躺在**另一个目录** `bin\` 里。
Windows 的 DLL 搜索只会看「exe 自身所在目录」，找不到跨目录的依赖，于是 `gcc` 直接退出码 1
且**不打印任何错误**，`cc` crate 只会报一句语焉不详的 `Compiler family detection failed`。

更麻烦的是 **cargo 的 `[env]` 救不了它**：cargo 在执行 build script 前会用父进程的 PATH
重新拼装搜索路径，配置文件里写的 `PATH` 会被直接丢弃（已实测验证）。

所以 `D:\msys64\mingw64\bin` 必须在**进程级** PATH 里 —— 这正是 `build.sh` 存在的理由：

```bash
export PATH="/d/msys64/mingw64/bin:$PATH"   # build.sh 自动做这件事
```

换机器时改掉上面三处路径即可（`build.sh` 也会自动在 `/d/msys64`、`/c/msys64`、`/c/mingw64`
里探测，或用 `MINGW_BIN=/path/to/mingw64/bin ./build.sh` 显式指定）。
若没装 MSYS2，可用便携版 [WinLibs](https://winlibs.com/) 解压即用。

## 配置

不带 `--config` 时按顺序查找，**先找到的生效**：

| 顺序 | 文件 | 用途 |
|---|---|---|
| 1 | `openworker.local.toml` | 放真实 API key，**已被 `.gitignore` 排除** |
| 2 | `openworker.toml` | 可提交的示例配置，不含密钥 |

显式传入的 `--config` 若文件不存在会**直接报错**（而不是悄悄回退到默认值）。
配置里的未知字段同样立即报错，`mdoel = "..."` 这类错字不会被静默吞掉。

```toml
[model]
provider = "deepseek"        # openai | deepseek | ollama | custom
api_key  = ""                # 留空则读环境变量
model    = "deepseek-v4-flash"
# base_url = "https://..."   # 任意 provider 都可覆盖端点（代理 / 网关 / vLLM / 自建）
```

API key 的查找顺序：`model.api_key` → provider 专属环境变量（`DEEPSEEK_API_KEY`）→
`OPENAI_API_KEY`。空字符串视为「未设置」，不会被当成有效密钥。

## 运行

```bash
# 单次提问（自动读取 openworker.local.toml）
./target/release/openworker run --prompt "读一下 README.md，给我三句话摘要"
# 交互式 REPL
./target/release/openworker run
# 指定模型 / 会话 / 模式
./target/release/openworker run --model deepseek-v4-pro --session demo --mode auto
# 显示模型思维链（默认隐藏）
./target/release/openworker run --show-reasoning --prompt "..."
```

`--mode auto` 无人值守全放行；默认的 `interactive` 会在写文件 / 执行命令前逐条询问。

推理模型（DeepSeek `reasoning_content`、o1 系列 `reasoning`）的思维链**默认不打印** ——
它通常比答案长得多，与正文混在一起会让人分不清哪句才是结论。加 `--show-reasoning`
可以看到，以暗色渲染区分。

本地 Ollama（零成本、完全离线）：把 `provider` 改为 `ollama` 即可，无需 API key。

MCP：

```bash
./target/release/openworker mcp list                 # 列出已配置 MCP 服务器的工具
./target/release/openworker mcp call filesystem read_file '{"path":"Cargo.toml"}'
```

自动化（cron 调度）：

```bash
./target/release/openworker automation list
./target/release/openworker automation run standup   # 立即跑一次
./target/release/openworker automation serve        # 启动调度循环，到点自动执行
```

## 测试

单元测试：

```bash
./build.sh test
```

端到端冒烟测试（**不需要 API key、不联网、不用真模型**）：

```bash
./smoke.sh --release
```

它会拉起两个本地夹具——`tests/fixtures/mock_llm.py`（OpenAI 兼容的假模型服务，第一轮返回
`list_dir` 工具调用、第二轮返回文本）和 `tests/fixtures/mock_mcp.py`（暴露一个 `echo` 工具的
stdio MCP 服务）——然后逐项断言：

```
  PASS  agent loop calls a tool          模型 → 工具调用
  PASS  agent loop finishes the turn     工具结果回灌 → 收尾
  PASS  MCP tool discovery               mcp__mock__echo
  PASS  MCP tool invocation              stdio JSON-RPC 往返
  PASS  automation listing               cron 解析
  PASS  config typos are rejected        配置错字立即报错
```

## 设计要点

- **模型层与厂商解耦**：运行时只认 `ProviderClient` 这个 trait。`OpenAICompatibleProvider` 用 OpenAI Chat Completions 线路，因此同一份代码能驱动 OpenAI、以及几乎所有 OpenAI 兼容端点（Ollama、DeepSeek、Qwen、GLM、Mistral…）。`base_url` 指向 `http://localhost:11434/v1` 即本地 Ollama。
- **流式 + 工具调用**：SSE 解析出 `text_delta` / `reasoning_delta`，并累积结构化 `tool_calls`；同时实现了“文本工具调用抢救”（部分本地模型把调用写成文本而不是 `tool_calls` 字段，这里会解析回来）。
- **Agent 循环**：一个用户回合 = 多次“模型↔工具”迭代，直到模型不再请求工具、或达到迭代上限。写文件 / 跑命令等高风险动作默认需要审批（`--mode auto` 可无人值守全放行）。读取类低风险工具并发执行，其余严格串行。
- **MCP 即工具**：任意 MCP 服务器启动后其工具自动成为 Agent 可调用的工具，无需改引擎代码。

## 与上游的差异 / 已知取舍

- 记忆用 JSONL 替代 SQLite（零 C 依赖、易检视）。`ChatMessage` 契约不变，可平滑换回 SQLite。
- 未实现：自动压缩（auto-compaction）、GUI/桌面壳、语音、25+ 连接器中的商业 SaaS 适配器（用 MCP 取代，更通用）、富事件总线与审计落盘。
- 这是“核心引擎”重写，不是字节级移植；目标是保留架构语义与开发者体验，而非逐行对应。

## License

与上游一致：MIT。
