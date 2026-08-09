//! Tools — ports `coworker/tools.py` (the `ToolRegistry`) plus the built-in local tools.
//!
//! A [`Tool`] is an async callable with an OpenAI-shaped schema. Tools execute inside the
//! engine's tool-handling step. Built-ins cover the local-first workflow: reading/writing
//! files, listing directories, running shell commands, fetching URLs, and asking the user.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};
use async_trait::async_trait;
use serde_json::{json, Value};

use crate::permissions::ToolMetadata;
use crate::provider::{FunctionSpec, ToolSpec};

/// A tool the agent can call.
#[async_trait]
pub trait Tool: Send + Sync {
    /// OpenAI-shaped tool spec.
    fn spec(&self) -> ToolSpec;
    /// Metadata for the permission engine (risk, approval requirements).
    fn metadata(&self) -> ToolMetadata {
        ToolMetadata::default()
    }
    /// Execute the tool with JSON-object arguments.
    async fn call(&self, args: Value) -> Result<Value>;
}

/// A registry of named tools.
#[derive(Clone)]
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        ToolRegistry {
            tools: HashMap::new(),
        }
    }

    pub fn register(&mut self, tool: Arc<dyn Tool>) {
        let name = tool.spec().function.name.clone();
        self.tools.insert(name, tool);
    }

    pub fn get(&self, name: &str) -> Option<&Arc<dyn Tool>> {
        self.tools.get(name)
    }

    /// Merge another registry's tools into this one.
    pub fn extend(&mut self, other: &ToolRegistry) {
        for (k, v) in &other.tools {
            self.tools.insert(k.clone(), v.clone());
        }
    }

    pub fn schemas(&self) -> Vec<ToolSpec> {
        self.tools.values().map(|t| t.spec()).collect()
    }

    pub fn names(&self) -> Vec<String> {
        self.tools.keys().cloned().collect()
    }

    pub async fn call(&self, name: &str, args: Value) -> Result<Value> {
        let tool = self
            .tools
            .get(name)
            .ok_or_else(|| anyhow!("unknown tool: {}", name))?;
        tool.call(args).await
    }

    pub fn len(&self) -> usize {
        self.tools.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Build a registry pre-populated with the built-in local tools.
pub fn register_builtins() -> ToolRegistry {
    let mut r = ToolRegistry::new();
    r.register(Arc::new(ReadFile));
    r.register(Arc::new(WriteFile));
    r.register(Arc::new(ListDir));
    r.register(Arc::new(RunCommand));
    r.register(Arc::new(WebFetch));
    r.register(Arc::new(AskUser::new()));
    r.register(Arc::new(crate::weather::GetWeather));
    r.register(Arc::new(crate::pdf::PdfToMarkdown));
    r
}

pub(crate) fn str_arg(args: &Value, key: &str) -> Result<String> {
    args.get(key)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow!("missing string argument '{}'", key))
}

// ---------------------------------------------------------------------------
// Built-in tools
// ---------------------------------------------------------------------------

/// Read a UTF-8 text file from the local filesystem.
pub struct ReadFile;
#[async_trait]
impl Tool for ReadFile {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            r#type: "function".into(),
            function: FunctionSpec {
                name: "read_file".into(),
                description: "Read a UTF-8 text file from the local filesystem and return its content. Relative paths resolve against the current working directory.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": { "path": { "type": "string", "description": "Path to the file to read" } },
                    "required": ["path"]
                }),
            },
        }
    }
    fn metadata(&self) -> ToolMetadata {
        ToolMetadata {
            risk_level: crate::permissions::RiskLevel::Low,
            ..Default::default()
        }
    }
    async fn call(&self, args: Value) -> Result<Value> {
        let path = str_arg(&args, "path")?;
        let content = tokio::fs::read_to_string(&path).await?;
        Ok(json!({ "path": path, "content": content }))
    }
}

/// Write text to a local file (creating or overwriting it).
pub struct WriteFile;
#[async_trait]
impl Tool for WriteFile {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            r#type: "function".into(),
            function: FunctionSpec {
                name: "write_file".into(),
                description: "Write text to a local file, creating parent directories as needed. Use this to produce deliverables (documents, reports, code).".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Destination path" },
                        "content": { "type": "string", "description": "Text content to write" }
                    },
                    "required": ["path", "content"]
                }),
            },
        }
    }
    fn metadata(&self) -> ToolMetadata {
        ToolMetadata {
            risk_level: crate::permissions::RiskLevel::High,
            requires_approval: true,
            capabilities: vec!["fs".into()],
            ..Default::default()
        }
    }
    async fn call(&self, args: Value) -> Result<Value> {
        let path = str_arg(&args, "path")?;
        let content = str_arg(&args, "content")?;
        if let Some(parent) = std::path::Path::new(&path).parent() {
            if !parent.as_os_str().is_empty() {
                tokio::fs::create_dir_all(parent).await?;
            }
        }
        tokio::fs::write(&path, content.as_bytes()).await?;
        Ok(json!({ "path": path, "bytes": content.len() }))
    }
}

/// List the entries of a directory.
pub struct ListDir;
#[async_trait]
impl Tool for ListDir {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            r#type: "function".into(),
            function: FunctionSpec {
                name: "list_dir".into(),
                description: "List files and subdirectories of a directory. Defaults to the current working directory when no path is given.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": { "path": { "type": "string", "description": "Directory to list (optional)" } }
                }),
            },
        }
    }
    fn metadata(&self) -> ToolMetadata {
        ToolMetadata {
            risk_level: crate::permissions::RiskLevel::Low,
            ..Default::default()
        }
    }
    async fn call(&self, args: Value) -> Result<Value> {
        let path = args
            .get("path")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| ".".to_string());
        let mut entries = Vec::new();
        let mut rd = tokio::fs::read_dir(&path).await?;
        while let Some(entry) = rd.next_entry().await? {
            let meta = entry.metadata().await?;
            let kind = if meta.is_dir() { "dir" } else { "file" };
            entries.push(json!({
                "name": entry.file_name().to_string_lossy(),
                "kind": kind,
                "size": meta.len()
            }));
        }
        entries.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));
        Ok(json!({ "path": path, "entries": entries }))
    }
}

/// Choose the shell used to run a command.
///
/// On Unix this is `sh -c`. On Windows we prefer a POSIX login shell (Git Bash or MSYS2
/// `bash -lc`) so the model's natural `grep`/`head`/`sed`/`find`/`git` commands actually
/// work — bare `cmd /C` has none of those and every Unix-style command fails with
/// "'grep' 不是内部或外部命令". We fall back to `cmd /C` only if no bash is found.
fn shell_for_command() -> (String, String) {
    #[cfg(windows)]
    {
        let candidates = [
            r"C:\Program Files\Git\bin\bash.exe",
            r"C:\Program Files\Git\usr\bin\bash.exe",
            r"D:\msys64\usr\bin\bash.exe",
            r"C:\msys64\usr\bin\bash.exe",
        ];
        for c in candidates {
            if std::path::Path::new(c).exists() {
                return (c.to_string(), "-lc".to_string());
            }
        }
        ("cmd".to_string(), "/C".to_string())
    }
    #[cfg(not(windows))]
    {
        ("sh".to_string(), "-c".to_string())
    }
}

/// Run a shell command and capture its output.
pub struct RunCommand;
#[async_trait]
impl Tool for RunCommand {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            r#type: "function".into(),
            function: FunctionSpec {
                name: "run_command".into(),
                description: "Execute a shell command on the local machine and return its combined stdout/stderr and exit code. Use for builds, git, tests, and other CLI tooling.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": { "command": { "type": "string", "description": "The command to run. On Unix it runs via `sh -c`; on Windows it runs inside Git Bash / MSYS2 `bash -lc` (so grep/head/sed/find/git work), falling back to `cmd /C`. Prefer portable Unix-style commands." } },
                    "required": ["command"]
                }),
            },
        }
    }
    fn metadata(&self) -> ToolMetadata {
        ToolMetadata {
            risk_level: crate::permissions::RiskLevel::High,
            requires_approval: true,
            capabilities: vec!["shell".into()],
            ..Default::default()
        }
    }
    async fn call(&self, args: Value) -> Result<Value> {
        let command = str_arg(&args, "command")?;
        let (shell, flag) = shell_for_command();
        let output = tokio::process::Command::new(&shell)
            .arg(&flag)
            .arg(&command)
            .output()
            .await?;
        let stdout = decode_console(&output.stdout);
        let stderr = decode_console(&output.stderr);
        Ok(json!({
            "command": command,
            "exit_code": output.status.code(),
            "stdout": stdout,
            "stderr": stderr
        }))
    }
}

/// Decode bytes written by a child process to stdout/stderr.
///
/// Modern tooling (and all of Unix) emits UTF-8, so that is tried first. Windows console
/// programs — including `cmd.exe`'s own error messages — instead use the OEM code page,
/// and decoding those bytes as UTF-8 turns every non-ASCII message into mojibake, which
/// then gets fed to the model as if it were the real error text.
fn decode_console(bytes: &[u8]) -> String {
    if let Ok(s) = std::str::from_utf8(bytes) {
        return s.to_string();
    }
    #[cfg(windows)]
    if let Some(enc) = oem_encoding() {
        return enc.decode(bytes).0.into_owned();
    }
    String::from_utf8_lossy(bytes).into_owned()
}

/// The console's OEM code page, mapped to a decoder. `None` for code pages we don't map,
/// which falls back to lossy UTF-8.
#[cfg(windows)]
fn oem_encoding() -> Option<&'static encoding_rs::Encoding> {
    let label: &str = match unsafe { windows_sys::Win32::Globalization::GetOEMCP() } {
        936 => "gbk",
        950 => "big5",
        932 => "shift_jis",
        949 => "euc-kr",
        866 => "ibm866",
        1250..=1258 => "windows-1252",
        65001 => "utf-8",
        _ => return None,
    };
    encoding_rs::Encoding::for_label(label.as_bytes())
}

/// Fetch a URL and return its body as text.
pub struct WebFetch;
#[async_trait]
impl Tool for WebFetch {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            r#type: "function".into(),
            function: FunctionSpec {
                name: "web_fetch".into(),
                description: "Fetch a URL over HTTP(S) and return the response body as text (truncated to 64 KiB).".into(),
                parameters: json!({
                    "type": "object",
                    "properties": { "url": { "type": "string", "description": "The URL to fetch" } },
                    "required": ["url"]
                }),
            },
        }
    }
    fn metadata(&self) -> ToolMetadata {
        ToolMetadata {
            risk_level: crate::permissions::RiskLevel::Low,
            ..Default::default()
        }
    }
    async fn call(&self, args: Value) -> Result<Value> {
        let url = str_arg(&args, "url")?;
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()?;
        let resp = client.get(&url).send().await?.error_for_status()?;
        let body = resp.text().await?;
        const LIMIT: usize = 64 * 1024;
        let truncated = if body.len() > LIMIT {
            // Back off to the nearest char boundary so we never slice mid-UTF-8.
            let mut end = LIMIT;
            while end > 0 && !body.is_char_boundary(end) {
                end -= 1;
            }
            format!("{}…[truncated]", &body[..end])
        } else {
            body
        };
        Ok(json!({ "url": url, "body": truncated, "bytes": truncated.len() }))
    }
}

/// Ask the user a question and return their answer.
///
/// The default construction blocks on the console (`stdin`), which is right for the CLI but
/// would deadlock the native GUI (no console input) and would hang unattended automations.
/// Surfaces inject an [`AskUserSink`] to redirect the question to their own UI / policy.
pub struct AskUser {
    sink: Option<Arc<dyn AskUserSink>>,
}

/// Where an `ask_user` question is delivered and answered. Blocking by design: the engine
/// thread waits for the answer, exactly like an approval prompt.
pub trait AskUserSink: Send + Sync {
    fn ask(&self, question: &str) -> String;
}

impl AskUser {
    /// Default: read the answer from the process console (CLI / REPL).
    pub fn new() -> Self {
        AskUser { sink: None }
    }
    /// Route the question to the given sink instead of the console (GUI / automation).
    pub fn with_sink(sink: Arc<dyn AskUserSink>) -> Self {
        AskUser { sink: Some(sink) }
    }
}

impl Default for AskUser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for AskUser {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            r#type: "function".into(),
            function: FunctionSpec {
                name: "ask_user".into(),
                description: "Ask the user a clarifying question and return their free-text answer. Use when you need a decision or input you cannot infer.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": { "question": { "type": "string", "description": "The question to ask" } },
                    "required": ["question"]
                }),
            },
        }
    }
    fn metadata(&self) -> ToolMetadata {
        ToolMetadata {
            risk_level: crate::permissions::RiskLevel::Low,
            ..Default::default()
        }
    }
    async fn call(&self, args: Value) -> Result<Value> {
        let question = str_arg(&args, "question")?;
        if let Some(sink) = &self.sink {
            // GUI / automation path: block on the injected sink, which owns its own reply
            // channel and resolves the answer when the user (or policy) replies.
            return Ok(json!({ "answer": sink.ask(&question) }));
        }
        use std::io::Write;
        println!("  ❓ {}", question);
        print!("    your answer: ");
        let _ = std::io::stdout().flush();
        let mut s = String::new();
        let _ = std::io::stdin().read_line(&mut s);
        Ok(json!({ "answer": s.trim().to_string() }))
    }
}

/// Record a durable note into the current session's recap, so future sessions start knowing it.
///
/// Not part of [`register_builtins`]: it needs to know which session is live, so the CLI/GUI
/// registers it once that is known. Deliberately low-risk and approval-free — it writes only to
/// OpenWorker's own recap directory, never to the user's project files.
pub struct Remember {
    store: crate::recall::RecallStore,
    session: String,
}

impl Remember {
    pub fn new(store: crate::recall::RecallStore, session: impl Into<String>) -> Self {
        Remember {
            store,
            session: session.into(),
        }
    }
}

#[async_trait]
impl Tool for Remember {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            r#type: "function".into(),
            function: FunctionSpec {
                name: "remember".into(),
                description: "把一条需要长期记住的事实写入本会话的记忆文件；未来开启新会话时，它会在对话开始前被自动读回。\
适合记录：用户明确表达的偏好与约定（工具链、命名、风格）、交付物的准确路径、关键决策及其理由、遗留问题与下一步。\
请在完成实质性工作后、或用户告诉你一个长期约定时调用。一次一条，写成一句自足的话（不要用「上面那个文件」这类指代），不要记录临时细节或密钥。"
                    .into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "note": { "type": "string", "description": "要长期记住的一条事实，一句自足的话" }
                    },
                    "required": ["note"]
                }),
            },
        }
    }
    fn metadata(&self) -> ToolMetadata {
        ToolMetadata {
            category: "memory".into(),
            risk_level: crate::permissions::RiskLevel::Low,
            requires_approval: false,
            capabilities: vec!["memory".into()],
        }
    }
    async fn call(&self, args: Value) -> Result<Value> {
        let note = str_arg(&args, "note")?;
        let path = self.store.append_note(&self.session, &note)?;
        Ok(json!({
            "ok": true,
            "session": self.session,
            "path": path.display().to_string(),
        }))
    }
}

/// Author a new skill (or update an existing one) and **hot-reload it into the live registry**
/// so the same turn can call it.
///
/// A skill is a small directory under `~/.openworker/skills/<name>/` (user scope) or
/// `./.openworker/skills/<name>/` (project scope). `content` is the full `SKILL.md` text —
/// YAML frontmatter declaring `name`, `description`, `risk`, `runtime`, plus optional
/// `entry` / `timeout_secs` / `parameters`, and a Markdown body shown to the model. The
/// referenced entry script must already exist (or `entry:` must point at a file you'll create
/// in the same call's follow-up). The skill becomes a regular tool from the moment this
/// returns, so the next model response can call `<name>` without restarting.
///
/// Not part of [`register_builtins`]: it needs the live `Arc<Mutex<ToolRegistry>>` to register
/// the new tool into, so the CLI/GUI wires it in once the engine exists.
pub struct WriteSkill {
    registry: std::sync::Arc<std::sync::Mutex<ToolRegistry>>,
}

impl WriteSkill {
    pub fn new(registry: std::sync::Arc<std::sync::Mutex<ToolRegistry>>) -> Self {
        WriteSkill { registry }
    }
}

#[async_trait]
impl Tool for WriteSkill {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            r#type: "function".into(),
            function: FunctionSpec {
                name: "write_skill".into(),
                description: "创建一个新的可调用 skill，或更新已有的 skill，立即在当前会话生效。\
skill 是放在 ~/.openworker/skills/<name>/ 或 <cwd>/.openworker/skills/<name>/ 下的一个目录：\
SKILL.md 是 YAML 清单+Markdown 说明，旁边放 entry 脚本（python/bash/node）。脚本通过 stdin 收到参数 JSON，\
把结果 JSON 打印到 stdout 即可。调用本工具后，你写的 skill 立刻变成可调用的函数，下一轮模型就能直接调用它。\
当遇到现成工具组合起来很麻烦、又会被反复用到的能力时，应当主动调用本工具把它沉淀成 skill。\
如果 entry 脚本还不存在，先用 write_file 把脚本本体写好，再调用本工具。"
                    .into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "skill 名称（仅小写字母/数字/下划线，长度 2-64，必须以字母开头）。后续将以该名作为函数名。"
                        },
                        "content": {
                            "type": "string",
                            "description": "完整 SKILL.md 文本（含 --- 包裹的 YAML frontmatter：name/description/risk/runtime/entry/parameters 均可选）"
                        },
                        "scope": {
                            "type": "string",
                            "enum": ["user", "project"],
                            "description": "保存位置。user=~/.openworker/skills（跨项目复用）；project=./.openworker/skills（随仓库走、可提交）。默认 user。"
                        },
                        "overwrite": {
                            "type": "boolean",
                            "description": "如果同名 skill 已存在，是否覆盖。默认 false（已存在则报错）。"
                        }
                    },
                    "required": ["name", "content"]
                }),
            },
        }
    }

    fn metadata(&self) -> ToolMetadata {
        ToolMetadata {
            category: "skill".into(),
            risk_level: crate::permissions::RiskLevel::Medium, // writes user-controlled code
            requires_approval: true,
            capabilities: vec!["file_write".into(), "skill".into()],
        }
    }

    async fn call(&self, args: Value) -> Result<Value> {
        let name = str_arg(&args, "name")?.trim().to_string();
        let content = str_arg(&args, "content")?.to_string();
        let scope = args
            .get("scope")
            .and_then(|v| v.as_str())
            .unwrap_or("user");
        let overwrite = args
            .get("overwrite")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // Validate the manifest first so we never half-write a broken skill to disk.
        let dir = match scope {
            "user" => crate::skills::user_skills_dir()
                .ok_or_else(|| anyhow::anyhow!("could not resolve the user home directory"))?,
            "project" => crate::skills::project_skills_dir(),
            other => bail!("scope must be `user` or `project`, got `{other}`"),
        };
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("create skills directory {}", dir.display()))?;
        let skill_dir = dir.join(&name);
        let manifest_path = skill_dir.join("SKILL.md");

        if manifest_path.exists() && !overwrite {
            bail!(
                "skill `{name}` already exists at {} (pass `overwrite: true` to replace it)",
                manifest_path.display()
            );
        }

        // Parse *in a temp dir* so we can validate the entry file path without committing yet.
        std::fs::create_dir_all(&skill_dir)
            .with_context(|| format!("create skill directory {}", skill_dir.display()))?;
        let parsed = crate::skills::parse_skill(&content, &skill_dir).with_context(|| {
            format!("validation failed for skill `{name}`")
        })?;
        if parsed.name != name {
            bail!(
                "the `name:` in SKILL.md (`{}`) does not match the `name` argument (`{}`)",
                parsed.name,
                name
            );
        }

        std::fs::write(&manifest_path, content)
            .with_context(|| format!("write {}", manifest_path.display()))?;

        // Hot-load: build a fresh SkillTool from the just-written manifest and register it.
        let tool: Arc<dyn Tool> = Arc::new(crate::skills::SkillTool::new(parsed));
        let status = {
            let mut r = self.registry.lock().unwrap();
            let was_present = r.get(name.as_str()).is_some();
            r.register(tool);
            if was_present {
                "updated"
            } else {
                "created"
            }
        };

        crate::logger::info(
            "skills",
            &format!("hot-reloaded `{name}` ({status}) -> {}", manifest_path.display()),
        );

        Ok(json!({
            "ok": true,
            "name": name,
            "status": status,
            "path": manifest_path.display().to_string(),
            "scope": scope,
            "note": "the new tool is available in the next model response; call it by name."
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utf8_output_is_decoded_verbatim() {
        assert_eq!(decode_console("héllo 世界".as_bytes()), "héllo 世界");
    }

    #[test]
    fn invalid_utf8_does_not_panic_or_lose_ascii() {
        // 0xB2 0xBB 0xCA 0xC7 is "不是" in GBK and invalid UTF-8. Whatever the host's code
        // page, decoding must not panic and must keep the surrounding ASCII intact — this
        // is the cmd.exe error-message path that used to surface as mojibake.
        let mut bytes = b"prog: ".to_vec();
        bytes.extend_from_slice(&[0xB2, 0xBB, 0xCA, 0xC7]);
        let out = decode_console(&bytes);
        assert!(out.starts_with("prog: "), "got {out:?}");
        assert!(!out.is_empty());
    }
}
