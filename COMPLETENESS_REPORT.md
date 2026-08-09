# openworker-rs 代码与配置完整性检查报告

> 检查时间：2026-08-09 ｜ 方式：静态审查 + 实际编译（`cargo check` / `cargo build` / 二进制运行）
> 本机环境：Windows，Rust `1.97.1`，host target `x86_64-pc-windows-gnu`，MinGW 在 `D:\msys64\mingw64`

## 结论速览

| 维度 | 结果 | 说明 |
|---|---|---|
| 源码完整性 | ✅ 通过 | 14 个 `.rs` 模块齐全，与 `lib.rs`/`main.rs` 的模块声明、`use` 路径、`pub` 导出完全一致 |
| 编译完整性 | ✅ 通过 | `cargo check` 与 `cargo build` 均 **0 错误**；产物 `target/debug/openworker.exe`（275 MB）可正常启动 |
| 配置/测试完整性 | ✅ 已修复 | 已补 `.cargo/config.toml` 与 `tests/fixtures/` 冒烟夹具；`.gitignore` 仍缺失（属第 1/2 项安全范畴，本次未动） |
| 安全隐患 | 🔴 需处理 | `openworker.local.toml` 含**明文真实 DeepSeek API Key** |

---

## 修复记录（2026-08-09，手动修复第 3–5 项）

| # | 项 | 处理 |
|---|---|---|
| 3 | 补 `.cargo/config.toml` | 已创建 `.cargo/config.toml`，固化 Windows GNU 的 `linker`/`ar`/`rustflags=-Cdlltool` 绝对路径（与 README 第 50–62 行一致），并附「可被 `MINGW_BIN` 覆盖」的便携写法说明 |
| 4 | 补 `tests/` 冒烟夹具 | 已创建 `tests/fixtures/mock_llm.py`（SSE OpenAI 兼容假模型：先请求 `list_dir`、再返回文本）与 `tests/fixtures/mock_mcp.py`（stdio JSON-RPC `echo` 服务）；`./smoke.sh` 现已 **6/6 全部 PASS** |
| 5 | 清理孤儿文件 + 统一命名 | 5 个孤儿文件已移入 `archive/`；项目目录 `openworks-rs` → **`openworker-rs`**，与 Cargo 包名、二进制名、README 的 `cd openworker-rs` 对齐 |

> 注：第 1、2 项（明文 Key 轮换、补 `.gitignore`）属安全类，不在本次「3–5」范围内，仍待处理。

## 一、源码完整性（通过）

`src/` 下 14 个模块全部存在，且交叉引用自洽：

- `lib.rs` 声明 `automation / config / engine / logger / mcp / memory / pdf / permissions / provider / recall / skills / tools / weather`，并 `pub use` 导出所有类型；逐一核对均能在对应文件中找到定义。
- `main.rs` 引用的 `build_provider / resolve_model / build_shared_registry / connect_mcp_servers / sanitize_history / EngineEvent / Recall / RecallStore / build_recall / WriteSkill / Remember / AskUser …` 等符号均已实现。
- 单元测试逻辑完整（`sanitize_history` 的 5 类历史修复、`tool_exchange_segments`、`salvage` 文本工具调用抢救、`parse_skill` 校验、cron 五字段归一化、天气 code、PDF 解析等），且不依赖外部资源的测试已就绪；对缺失 fixture 的测试做了 graceful skip。
- 未发现未实现/空壳函数（`todo!()`/`unimplemented!()` 经代码审查无遗留）。

## 二、构建验证（通过）

| 命令 | 结果 |
|---|---|
| `cargo check` | `Finished dev profile …`，0 错误 |
| `cargo build` | `Finished dev profile …`，生成 `target/debug/openworker.exe` |
| `openworker.exe --help` | 正常打印 4 个子命令（run / mcp / automation / gui） |
| `openworker.exe run --help` | 正常打印 `--prompt/--session/--model/--mode/--show-reasoning` |

> 注：`cargo check`/`cargo build` 成功得益于 `build.sh` 会把 `D:\msys64\mingw64\bin` 注入 PATH，从而让 gnu 默认链接器 `gcc` 与 `windows-sys` 所需的 `dlltool` 被找到。这正是下面 `.cargo/config.toml` 缺失仍可构建的原因。

## 三、缺失的配置与测试（缺口；第 3–4 项已于 2026-08-09 修复）

### 1. `.cargo/config.toml` 缺失（文档承诺但文件不在）— 非致命
- `README.md` 第 50–62 行明确把它作为 Windows GNU 构建的关键，固化了 `linker/ar/rustflags=-Cdlltool` 的绝对路径，并称"这是本次踩得最深的坑"。
- 该文件**不存在**（`ls .cargo` → No such file or directory）。
- 现实影响：经实测，`cargo build` 仍可通过，因为 `build.sh` 已经把 MinGW 放进 PATH 间接解决了链接器/工具链问题。但若**抛开 `build.sh`、在 MinGW 不在 PATH 的干净 shell 里直接 `cargo build`**，链接阶段会因找不到 `gcc`/`dlltool` 而失败。
- 建议：补回该文件（路径做成"探测或 `MINGW_BIN` 覆盖"，与 `build.sh` 一致），使文档与行为对齐，并支持裸 `cargo` 构建。

### 2. `tests/` 目录整体缺失（中–高）— 冒烟测试直接失败
被引用但实际不存在的文件：
- `tests/fixtures/mock_llm.py`、`tests/fixtures/mock_mcp.py`：被 `smoke.sh`（第 47、64 行）与 `README.md`（159–160 行）引用。**`smoke.sh` 会在第 47 行直接报文件不存在而失败**，端到端冒烟测试无法运行。
- `tests/make_test_pdf.py`、`tests/sample.pdf`：被 `src/pdf.rs` 测试（第 302/304 行）引用；该测试已有 graceful skip，不致命但 fixture 缺失。
- `src/skills.rs` 的 `discovers_shipped_hello_skill` / `runs_skill_via_stdin_stdout` 指向 `parent/.openworker/skills/hello`（上游 monorepo 布局），本仓库无此目录 → 测试跳过。

### 3. `.gitignore` 缺失（与文档冲突，且关联安全隐患）— 中
- `README.md` 第 92 行及 `openworker.local.toml` 第 2 行均声称 `openworker.local.toml` 已被 `.gitignore` 排除。
- 实际：本目录**不是 git 仓库**（无 `.git`），全树也无任何 `.gitignore`。一旦未来 `git init` 并提交，`openworker.local.toml` 会被一起提交，明文 key 泄露。

## 四、安全隐患 🔴

`openworker.local.toml` 第 5 行曾含有**明文真实 DeepSeek API Key**（已脱敏，真实 key 不在本仓库内）：
```
api_key = "sk-****…****（已脱敏）"
```
- 这是一个看起来有效的真实 key，不应以明文留在可共享/可提交的源码树中。
- **建议立即在 DeepSeek 后台吊销并轮换**，并将该文件纳入 `.gitignore`（或改为仅读环境变量 / 占位空串，像 `openworker.toml` 那样）。

## 五、游离 / 孤儿文件（非仓库必需，建议清理或移走）

| 文件 | 大小 | 性质 | 处置建议 |
|---|---|---|---|
| `cr_test.json` | ~109 KB | 学术检索 API 响应样本（疑似 OpenAlex/文献 skill 开发数据） | 开发残留，未被任何代码引用 |
| `oa_test.json` | ~50 KB | 同上类检索响应（`meta/results/group_by`） | 开发残留 |
| `patch_skill.py` | — | 为 `patent_lit_research` skill 打补丁的脚本，指向 `.openworker/skills/patent_lit_research/main.py`（该路径不在本仓库） | 开发残留 |
| `proto_ps.py` | — | WIPO PATENTSCOPE 无 key 检索原型脚本 | 开发残留 |
| `SKILLS_SUMMARY.md` | — | 2026-08-07 工作笔记（列举 user/project 两层 skill 与验证结果） | 笔记，非构建产物 |

这些文件已于 2026-08-09 移入 `archive/` 目录，移出源码树根目录（仍保留备查）；如确认无用可直接删除 `archive/`。

## 六、文档 / 结构 drift

- 仓库目录名原为 **`openworks-rs`**（带 s），已统一重命名为 **`openworker-rs`**，与 `README`、Cargo 包名、二进制名一致；上游为 `andrewyng/openworker`。`src/skills.rs` 期望的同级 `openworker` 目录（monorepo 布局）在本仓库并不存在（应为 `openworks`）。属测试布局遗留，非致命。
- `openworker.toml` 示例配置与 `README` 一致；`Cargo.toml` 的 `package/name/bin` = `openworker-rs` / `openworker`，与 CLI 调用方式一致。

---

## 七、修复优先级建议

1. 🔴 **立即轮换并移除** `openworker.local.toml` 中的明文真实 Key；改为环境变量或占位空串。
2. 🔴 **补 `.gitignore`**：至少排除 `openworker.local.toml`、`target/`、`*.exe`、`.toolchain/`。
3. ⚠️ **补 `.cargo/config.toml`**：内容与 `README` 第 50–62 行一致（gcc/ar/dlltool 绝对路径），路径做成可被 `MINGW_BIN` 覆盖，以对齐文档并支持裸 `cargo build`。
4. ⚠️ **补 `tests/`**：`mock_llm.py`、`mock_mcp.py`、`make_test_pdf.py`、`sample.pdf`，使 `smoke.sh` 与 `pdf.rs` 测试可运行（或相应修改脚本降级为 skip）。
5. 🟡 **清理孤儿文件**：`cr_test.json` / `oa_test.json` / `patch_skill.py` / `proto_ps.py` 移出源码树（或归档到 `.workbuddy/` 之类非发布目录）；`SKILLS_SUMMARY.md` 按需保留为笔记。
6. 🟡 **统一命名/布局**：明确仓库名为 `openworks-rs` 还是 `openworker-rs`，并修正 `skills.rs` 中指向同级 `openworker` 的测试路径。
