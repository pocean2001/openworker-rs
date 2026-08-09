//! `openworker-rs` — a Rust rewrite of the OpenWorker core agent engine.
//!
//! This crate ports the heart of Andrew Ng's [OpenWorker](https://github.com/andrewyng/openworker)
//! Python backend (`coworker/`) to Rust:
//!
//! * [`provider`] — a provider-agnostic model layer (`OpenAICompatibleProvider` speaks the
//!   OpenAI Chat Completions wire format, which Ollama and most compat vendors also implement).
//! * [`engine`] — the owned agent loop: model ⇄ tools, streaming deltas, approvals, low-risk
//!   tool concurrency.
//! * [`tools`] — a `ToolRegistry` plus built-in tools (read/write file, list dir, run command,
//!   web fetch, ask user, write skill).
//! * [`skills`] — file-based, agent-authorable skills: drop a `SKILL.md` + script into
//!   `~/.openworker/skills/<name>/` or `./.openworker/skills/<name>/` and the agent can call it.
//! * [`weather`] — a weather skill (`get_weather`) powered by the free, keyless Open-Meteo API.
//! * [`mcp`] — a small stdio JSON-RPC 2.0 MCP client that turns any MCP server's tools into
//!   registry tools.
//! * [`memory`] — a dependency-free JSONL conversation store.
//! * [`recall`] — cross-session recall: per-session Markdown recaps, read back before a turn.
//! * [`automation`] — a cron-driven scheduler.
//! * [`permissions`] / [`config`] — approval policy and TOML configuration.

pub mod automation;
pub mod config;
pub mod engine;
pub mod logger;
pub mod mcp;
pub mod memory;
pub mod pdf;
pub mod permissions;
pub mod provider;
pub mod recall;
pub mod skills;
pub mod tools;
pub mod weather;

pub use automation::{Automation, AutomationRunner, Scheduler};
pub use config::{load_config, Config};
pub use engine::{sanitize_history, EngineEvent, Recall, TurnEngine};
pub use mcp::{connect_mcp_servers, tool_name, McpClient, McpServerDef, McpToolInfo};
pub use memory::MemoryStore;
pub use permissions::{
    ApprovalOutcome, Approver, AutoApprover, ConsoleApprover, Decision, DenyApprover,
    Mode, PermissionEngine, PermissionRequest, RiskLevel, ToolMetadata,
};
pub use recall::{RecallStore, RECALL_HEADER};
pub use skills::{
    default_search_roots, discover_skills, parse_skill, project_skills_dir, user_skills_dir,
    Skill, SkillTool, RESERVED_SKILL_NAMES,
};
pub use pdf::PdfToMarkdown;
pub use provider::{
    AssistantTurn, ChatMessage, CompletionRequest, FunctionSpec, ModelCapabilities, ModelSettings,
    OpenAICompatibleProvider, ProviderClient, StreamChunk, ToolCall, ToolSpec, TokenUsage,
};
pub use tools::{
    register_builtins, AskUser, AskUserSink, ListDir, ReadFile, Remember, RunCommand, Tool,
    ToolRegistry, WebFetch, WriteFile, WriteSkill,
};
pub use weather::GetWeather;
