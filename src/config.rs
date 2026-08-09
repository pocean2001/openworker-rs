//! TOML configuration.
//!
//! ```toml
//! [model]
//! provider = "openai"          # openai | ollama | custom
//! api_key  = "sk-..."          # or set OPENAI_API_KEY in the environment
//! model    = "gpt-4o-mini"
//! # base_url = "https://..."   # overrides the provider default (proxy, gateway,
//!                              # self-hosted endpoint); required for "custom"
//!
//! [[mcp_servers]]
//! name = "fs"
//! command = "npx"
//! args = ["-y", "@modelcontextprotocol/server-filesystem", "."]
//!
//! [[automations]]
//! name = "morning-brief"
//! prompt = "Summarize what changed in this repo since yesterday."
//! cron = "0 9 * * 1-5"
//! ```

use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

use crate::automation::Automation;
use crate::mcp::McpServerDef;

/// `deny_unknown_fields` on purpose: a silently-ignored typo in a config file is one of
/// the most annoying failure modes there is (`mdoel = "gpt-4o"` and you get the default
/// with no hint why). Better to fail loudly at startup.
#[derive(Debug, Clone, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    #[serde(default)]
    pub provider: String,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub base_url: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(default)]
    pub model: ModelConfig,
    #[serde(default)]
    pub mcp_servers: Vec<McpServerDef>,
    #[serde(default)]
    pub automations: Vec<Automation>,
    #[serde(default)]
    pub instructions: Option<String>,
    #[serde(default)]
    pub mode: Option<String>,
    /// Enable proactive auto-compaction of older history (default: on). Set to false to disable.
    #[serde(default)]
    pub auto_compress: Option<bool>,
    /// Fraction of the context budget (0..1) at which compaction triggers. Default 0.6.
    #[serde(default)]
    pub context_compress_ratio: Option<f64>,
    /// Max tool-call rounds allowed in a single turn. Default 30. Raise it for long build/test
    /// workflows; it is a runaway-loop guard, not a task-complexity limit.
    #[serde(default)]
    pub max_iterations: Option<u32>,
    /// Read earlier sessions' recaps into context before each turn (default: on).
    #[serde(default)]
    pub session_recall: Option<bool>,
    /// How many earlier sessions to recall. Default 3.
    #[serde(default)]
    pub session_recall_count: Option<usize>,
    /// Character ceiling on the injected recall block. Default 4000.
    #[serde(default)]
    pub session_recall_chars: Option<usize>,
}

impl Config {
    /// A reasonable default system prompt when the config doesn't supply one.
    pub fn effective_instructions(&self) -> String {
        self.instructions.clone().unwrap_or_else(|| {
            "You are OpenWorker, a local-first AI coworker. You deliver finished work, not just \
chat: documents, reports, code, and answers grounded in the user's own files and tools. When a \
task needs a file, write it. When you need information, use the tools available to you. Before \
any destructive or externally-visible action, prefer to confirm with the user.

Operating principles:

1. Diagnose before retrying. When a tool call fails or returns an unexpected result, read the \
error carefully and identify the root cause — environment (missing binary or wrong PATH), bad \
arguments, insufficient permissions, or a wrong assumption about the data. Do NOT blindly repeat \
the exact same failing command. Choose the smallest next action that disambiguates the failure, \
or ask the user for the missing fact.

2. Decompose complex tasks. For any multi-step request, first state a short plan (ordered steps \
and their dependencies), then execute one step at a time and verify each step before moving on. \
Prefer producing a concrete intermediate artifact over a long explanation. If a step can be \
checked (run a test, read a file, grep output), check it.

3. Respect the context window. Conversations accumulate tokens, and long tool outputs plus \
repeated full-file reads burn the budget fast. Prefer targeted reads (specific lines/ranges) over \
dumping whole large files, and read a file once then reason from what you kept. When the exchange \
gets long, compress: replace verbose back-and-forth with a one-line summary of what was decided and \
what remains, and drop redundant detail. If the task is genuinely open-ended, suggest continuing in \
a fresh session rather than letting context overflow.

4. Carry knowledge across sessions. A block marked [跨会话记忆] may be prepended to the \
conversation: it is a recap of your earlier sessions, provided as background — the user's real \
request is always the latest user message, and if the recap contradicts the present situation, the \
present situation wins. Treat remembered file paths and conclusions as leads to verify with a tool, \
not as established fact. When you finish a substantive piece of work, or the user states a lasting \
convention or preference, call the `remember` tool to record it as one self-contained sentence so \
your next session starts already knowing it. Do not record transient details or secrets.

5. Author reusable skills when the work recurs. A **skill** is a small directory (an SKILL.md \
manifest plus a script) under `~/.openworker/skills/<name>/` or `./.openworker/skills/<name>/` \
that becomes a callable function for you. When you find yourself stitching the same multi-tool \
dance together more than once, or the user asks for a capability that obviously belongs as a \
named tool (a `translate` skill, a `summarize_pdf` skill, a `query_db` skill, …), call the \
`write_skill` tool to author one. Each call takes the SKILL.md content + a scope and writes \
it to disk; the new tool is hot-loaded into your current session's registry the moment the \
call returns, so the next time the user asks for it you can just call `<name>(...)`. Drop the \
entry script with `write_file` first, then call `write_skill` so the manifest can point at it."
            .to_string()
        })
    }
}

pub fn load_config(path: &Path) -> Result<Config> {
    let txt = std::fs::read_to_string(path)
        .with_context(|| format!("read config {}", path.display()))?;
    let cfg: Config = toml::from_str(&txt).with_context(|| "parse TOML config")?;
    Ok(cfg)
}
