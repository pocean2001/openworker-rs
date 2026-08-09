//! `openworker` — CLI for the Rust OpenWorker engine.
//!
//! Subcommands:
//!   run            Chat with the agent (single `--prompt` or an interactive REPL)
//!   mcp list       List tools exposed by configured MCP servers
//!   mcp call       Call an MCP tool directly
//!   automation …   List / run-once / serve scheduled tasks

use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

// Native GUI (the `gui` subcommand). Pure-Rust egui/eframe window — a real standalone
// desktop client: no browser, no local server, no WebView runtime. Ships as one binary.
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

use eframe::egui;
use serde::Serialize;
use tokio::runtime::{Builder, Handle};
use tokio::sync::mpsc;

use openworker_rs::*;
use openworker_rs::{ChatMessage, CompletionRequest, ModelSettings, OpenAICompatibleProvider, ProviderClient};

mod logger;

#[derive(Parser)]
#[command(name = "openworker", about = "Rust rewrite of the OpenWorker core agent engine")]
struct Cli {
    /// Path to the TOML config file.
    /// Defaults to ./openworker.local.toml, then ./openworker.toml, if either exists.
    #[arg(long)]
    config: Option<PathBuf>,
    /// Also print the model's chain-of-thought (DeepSeek `reasoning_content`, o1-style
    /// `reasoning`). Hidden by default: on reasoning models it is far longer than the
    /// answer and, rendered identically, becomes indistinguishable from it.
    #[arg(long, global = true)]
    show_reasoning: bool,
    #[command(subcommand)]
    command: Commands,
}

/// Set once from the CLI flag; read by the event renderer, which has no access to `Cli`.
static SHOW_REASONING: AtomicBool = AtomicBool::new(false);
/// True while an unterminated dim-styled reasoning run is on screen. Reasoning arrives one
/// token at a time, so wrapping each delta individually would emit two escape sequences per
/// token; instead we open the style once and close it when normal output resumes.
static DIM_OPEN: AtomicBool = AtomicBool::new(false);

/// Close the dim style if a reasoning run is currently open.
fn end_dim() {
    if DIM_OPEN.swap(false, Ordering::Relaxed) {
        print!("\x1b[0m");
    }
}

#[derive(Subcommand)]
enum Commands {
    /// Chat with the agent (single prompt or interactive REPL).
    Run {
        #[arg(long)]
        prompt: Option<String>,
        #[arg(long, default_value = "default")]
        session: String,
        #[arg(long)]
        model: Option<String>,
        #[arg(long, default_value = "interactive")]
        mode: String,
    },
    /// Inspect and call MCP servers.
    Mcp {
        #[command(subcommand)]
        action: McpAction,
    },
    /// Scheduled automations.
    Automation {
        #[command(subcommand)]
        action: AutomationAction,
    },
    /// Launch the standalone native GUI client (a desktop window).
    Gui,
}

#[derive(Subcommand)]
enum McpAction {
    /// List tools from every configured MCP server.
    List,
    /// Call one MCP tool by server + tool name.
    Call {
        server: String,
        tool: String,
        #[arg(default_value = "{}")]
        args: String,
    },
}

#[derive(Subcommand)]
enum AutomationAction {
    /// Print configured automations.
    List,
    /// Run one automation immediately.
    Run { name: String },
    /// Run the scheduler loop (fires due automations forever).
    Serve,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    logger::init_logger();
    let sub = match &cli.command {
        Commands::Run { .. } => "run",
        Commands::Mcp { .. } => "mcp",
        Commands::Automation { .. } => "automation",
        Commands::Gui => "gui",
    };
    logger::info(
        "main",
        &format!("launch subcommand={sub} config={:?}", cli.config),
    );
    SHOW_REASONING.store(cli.show_reasoning, Ordering::Relaxed);
    let cfg = match &cli.config {
        // Explicitly passed: a missing file is a hard error. Falling back to defaults here
        // would turn a mistyped path into a baffling "OPENAI_API_KEY not set" further down.
        Some(path) => load_config(path)?,
        // Implicit: the default config is optional, so absence is fine. `.local.toml` wins so
        // real credentials live in a gitignored file while `openworker.toml` stays a shareable
        // sample that can be committed.
        None => ["openworker.local.toml", "openworker.toml"]
            .iter()
            .map(PathBuf::from)
            .find(|p| p.exists())
            .map(|p| load_config(&p))
            .transpose()?
            .unwrap_or_default(),
    };

    match &cli.command {
        Commands::Run {
            prompt,
            session,
            model,
            mode,
        } => {
            cmd_run(&cfg, prompt.clone(), session, model.clone(), mode).await
        }
        Commands::Mcp { action } => cmd_mcp(&cfg, action).await,
        Commands::Automation { action } => cmd_automation(&cfg, action).await,
        Commands::Gui => cmd_gui(&cfg).await,
    }
}

// ---------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------

async fn cmd_run(
    cfg: &Config,
    prompt: Option<String>,
    session: &str,
    model_override: Option<String>,
    mode: &str,
) -> Result<()> {
    let provider = build_provider(cfg, model_override.as_deref())?;
    logger::info(
        "run",
        &format!(
            "CLI run session={session} model_override={model_override:?} mode={mode}",
        ),
    );
    let model = resolve_model(cfg, model_override.as_deref());

    // Cross-session recall. The store is shared by the engine (which reads recaps before every
    // turn) and the `remember` tool (which appends to the live session's recap).
    let recall_store = if cfg.session_recall.unwrap_or(true) {
        data_dir().ok().and_then(|d| RecallStore::new(&d).ok())
    } else {
        None
    };
    let registry = build_shared_registry(cfg, session, true, None).await?;

    let mode = Mode::from_str(mode);
    let approver: Box<dyn Approver> = if mode == Mode::Auto {
        Box::new(AutoApprover)
    } else {
        Box::new(ConsoleApprover)
    };
    let perms = PermissionEngine::new(mode);
    let mut engine = TurnEngine::new_shared(
        provider,
        registry,
        perms,
        model,
        Some(cfg.effective_instructions()),
        approver,
    );

    engine.set_auto_compress(cfg.auto_compress.unwrap_or(true));
    if let Some(r) = cfg.context_compress_ratio {
        engine.set_auto_compress_ratio(r);
    }
    if let Some(n) = cfg.max_iterations {
        engine.set_max_iterations(n);
    }
    if let Some(rs) = recall_store {
        engine.set_recall(build_recall(cfg, rs, session));
    }

    let store = MemoryStore::new(&data_dir()?)?;
    engine.load_history(store.load(session)?);

    match prompt {
        Some(p) => run_one(&mut engine, &store, session, p).await?,
        None => {
            println!("OpenWorker-rs REPL — 输入消息开始对话，exit / quit 退出。");
            loop {
                print!("你> ");
                let _ = io::stdout().flush();
                let mut line = String::new();
                if io::stdin().read_line(&mut line).is_err() {
                    break;
                }
                let line = line.trim();
                if line == "exit" || line == "quit" {
                    break;
                }
                if line.is_empty() {
                    continue;
                }
                run_one(&mut engine, &store, session, line.to_string()).await?;
            }
        }
    }
    Ok(())
}

async fn run_one(
    engine: &mut TurnEngine,
    store: &MemoryStore,
    session: &str,
    prompt: String,
) -> Result<()> {
    logger::info(
        "run",
        &format!("turn start session={session} prompt_len={}", prompt.chars().count()),
    );
    let mut sink = |ev: EngineEvent| {
        log_engine_event(&ev);
        emit_pretty(&ev);
    };
    engine.run_turn(prompt, &mut sink).await?;
    // Guard against persisting a dangling `assistant(tool_calls)` tail (e.g. after an
    // iteration-limit stop mid-tool-call): the next turn would get a 400 from the API.
    let hist = sanitize_history(engine.history().to_vec());
    store.save(session, &hist)?;
    logger::info("run", &format!("turn saved session={session}"));
    Ok(())
}

// ---------------------------------------------------------------------------
// MCP
// ---------------------------------------------------------------------------

async fn cmd_mcp(cfg: &Config, action: &McpAction) -> Result<()> {
    match action {
        McpAction::List => {
            if cfg.mcp_servers.is_empty() {
                println!("No MCP servers configured.");
                return Ok(());
            }
            let reg = connect_mcp_servers(&cfg.mcp_servers).await?;
            println!("MCP tools ({}):", reg.len());
            for spec in reg.schemas() {
                println!("  - {}: {}", spec.function.name, spec.function.description);
            }
        }
        McpAction::Call {
            server,
            tool,
            args,
        } => {
            let def = cfg
                .mcp_servers
                .iter()
                .find(|d| &d.name == server)
                .ok_or_else(|| anyhow!("no such MCP server: {}", server))?;
            let client = McpClient::connect(def).await?;
            let tools = client.list_tools().await?;
            let info = tools
                .iter()
                .find(|t| &t.name == tool)
                .ok_or_else(|| anyhow!("no such tool on server {}: {}", server, tool))?;
            let args: Value = serde_json::from_str(args).unwrap_or(json!({}));
            let result = client.call_tool(&info.name, args).await?;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Automations
// ---------------------------------------------------------------------------

async fn cmd_automation(cfg: &Config, action: &AutomationAction) -> Result<()> {
    match action {
        AutomationAction::List => {
            if cfg.automations.is_empty() {
                println!("No automations configured.");
            }
            for a in &cfg.automations {
                println!("- {}  (cron: {})  {}", a.name, a.cron, a.prompt);
            }
        }
        AutomationAction::Run { name } => {
            let auto = cfg
                .automations
                .iter()
                .find(|a| &a.name == name)
                .ok_or_else(|| anyhow!("no such automation: {}", name))?;
            let runner = build_runner(cfg)?;
            runner.run(&auto.prompt, auto.session.as_deref()).await;
        }
        AutomationAction::Serve => {
            let runner = build_runner(cfg)?;
            Scheduler::serve(cfg.automations.clone(), Arc::new(runner)).await?;
        }
    }
    Ok(())
}

struct AgentRunner {
    provider: Arc<dyn ProviderClient>,
    registry: Arc<Mutex<ToolRegistry>>,
    instructions: String,
    model: String,
}

#[async_trait::async_trait]
impl AutomationRunner for AgentRunner {
    async fn run(&self, prompt: &str, _session: Option<&str>) {
        let perms = PermissionEngine::new(Mode::Auto);
        let approver: Box<dyn Approver> = Box::new(AutoApprover);
        let mut engine = TurnEngine::new_shared(
            self.provider.clone(),
            self.registry.clone(),
            perms,
            self.model.clone(),
            Some(self.instructions.clone()),
            approver,
        );
        let mut sink = |ev: EngineEvent| emit_pretty(&ev);
        if let Err(e) = engine.run_turn(prompt.to_string(), &mut sink).await {
            eprintln!("automation error: {e}");
        }
    }
}

fn build_runner(cfg: &Config) -> Result<AgentRunner> {
    // Synchronous registry build: automations run unattended so we don't need write_skill.
    // (Filesystem-based skills are still loaded — they're just read-only here.)
    let reg = std::sync::Arc::new(std::sync::Mutex::new(register_builtins()));
    {
        let mut r = reg.lock().unwrap();
        // `ask_user` in an unattended automation would block forever on a console nobody is
        // reading — answer it deterministically instead and let the model continue on its own.
        let auto_sink: Arc<dyn AskUserSink> = Arc::new(AutoAskUserSink);
        r.register(std::sync::Arc::new(AskUser::with_sink(auto_sink)));
        let skills = skills::discover_skills(&skills::default_search_roots());
        for s in skills {
            let name = s.name.clone();
            r.register(std::sync::Arc::new(skills::SkillTool::new(s)));
            logger::info("registry", &format!("skill `{name}` registered (automation)"));
        }
    }
    let provider = build_provider(cfg, None)?;
    let model = resolve_model(cfg, None);
    Ok(AgentRunner {
        provider,
        registry: reg,
        instructions: cfg.effective_instructions(),
        model,
    })
}

/// Automation-only sink: no user is present, so any `ask_user` is answered with a fixed
/// "proceed on your best judgment" so the turn can complete unattended.
struct AutoAskUserSink;
impl AskUserSink for AutoAskUserSink {
    fn ask(&self, _question: &str) -> String {
        logger::warn(
            "automation",
            "ask_user called in unattended automation; answering with the default 'proceed'",
        );
        "（自动化运行，无用户在线；请基于现有信息自行决定，不要等待更多输入）".to_string()
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn resolve_model(cfg: &Config, override_model: Option<&str>) -> String {
    let provider = if cfg.model.provider.is_empty() {
        "openai"
    } else {
        cfg.model.provider.as_str()
    };
    override_model
        .map(|s| s.to_string())
        .or_else(|| cfg.model.model.clone())
        .unwrap_or_else(|| {
            match provider {
                "ollama" => "llama3",
                "deepseek" => "deepseek-v4-flash",
                _ => "gpt-4o-mini",
            }
            .to_string()
        })
}

fn build_provider(cfg: &Config, override_model: Option<&str>) -> Result<Arc<dyn ProviderClient>> {
    let provider = if cfg.model.provider.is_empty() {
        "openai"
    } else {
        cfg.model.provider.as_str()
    };
    // An empty string in the config is a placeholder, not a real key — treat it as unset
    // so we still fall back to the environment.
    let nonempty = |s: String| if s.trim().is_empty() { None } else { Some(s) };
    // Provider-specific env var first (DEEPSEEK_API_KEY etc.), then the generic one, so a
    // machine can hold keys for several backends at once without them clobbering each other.
    let env_var = match provider {
        "deepseek" => "DEEPSEEK_API_KEY",
        _ => "OPENAI_API_KEY",
    };
    let api_key = cfg
        .model
        .api_key
        .clone()
        .and_then(nonempty)
        .or_else(|| std::env::var(env_var).ok().and_then(nonempty))
        .or_else(|| std::env::var("OPENAI_API_KEY").ok().and_then(nonempty));
    // An explicit base_url always wins, whatever the provider — that is how you point
    // "openai" at a proxy/gateway, or "ollama" at a non-default host.
    let explicit_base = cfg.model.base_url.clone().and_then(nonempty);
    let model = resolve_model(cfg, override_model);
    let (base_url, key) = match provider {
        "ollama" => (
            explicit_base.unwrap_or_else(|| "http://localhost:11434/v1".to_string()),
            api_key.unwrap_or_else(|| "ollama".to_string()),
        ),
        "openai" => (
            explicit_base.unwrap_or_else(|| "https://api.openai.com/v1".to_string()),
            api_key.ok_or_else(|| {
                anyhow!("OPENAI_API_KEY not set (set env var or model.api_key in config)")
            })?,
        ),
        "deepseek" => (
            explicit_base.unwrap_or_else(|| "https://api.deepseek.com/v1".to_string()),
            api_key.ok_or_else(|| {
                anyhow!("DEEPSEEK_API_KEY not set (set env var or model.api_key in config)")
            })?,
        ),
        "custom" => (
            explicit_base
                .ok_or_else(|| anyhow!("model.base_url is required for provider = \"custom\""))?,
            api_key.ok_or_else(|| anyhow!("model.api_key is required for provider = \"custom\""))?,
        ),
        other => anyhow::bail!(
            "unknown model.provider '{}' (expected: openai | deepseek | ollama | custom)",
            other
        ),
    };
    let p = OpenAICompatibleProvider::with_base_url(&key, &base_url, &model);
    logger::info(
        "config",
        &format!(
            "provider={provider} model={model} base_url={base_url} api_key_set(len={})",
            key.len()
        ),
    );
    Ok(Arc::from(Box::new(p) as Box<dyn ProviderClient>))
}

/// 各服务商的默认 OpenAI 兼容 base_url。
fn default_base_url(provider: &str) -> String {
    match provider {
        "openai" => "https://api.openai.com/v1".to_string(),
        "ollama" => "http://localhost:11434/v1".to_string(),
        "deepseek" => "https://api.deepseek.com/v1".to_string(),
        _ => String::new(),
    }
}

/// 各服务商的默认模型名。
fn default_model(provider: &str) -> String {
    match provider {
        "ollama" => "llama3".to_string(),
        "deepseek" => "deepseek-v4-flash".to_string(),
        "openai" => "gpt-4o-mini".to_string(),
        _ => String::new(),
    }
}

/// Apply the config's recall overrides on top of the engine defaults.
fn build_recall(cfg: &Config, store: RecallStore, session: &str) -> Recall {
    let mut r = Recall::new(store, session);
    if let Some(n) = cfg.session_recall_count {
        r.sessions = n;
    }
    if let Some(n) = cfg.session_recall_chars {
        r.max_chars = n;
    }
    r
}

fn data_dir() -> Result<PathBuf> {
    let base = dirs::data_local_dir().unwrap_or_else(|| PathBuf::from("."));
    Ok(base.join("openworker-rs"))
}

/// Build the live, hot-reloadable `Arc<Mutex<ToolRegistry>>` shared by the engine and any
/// tool that needs to mutate it (currently just `write_skill`).
///
/// Loads in this order, last-wins per-name:
/// 1. Built-in tools (`read_file`, `write_file`, `run_command`, …)
/// 2. MCP tools (if any are configured)
/// 3. File-based skills (user-root first, project-root second; project shadows user)
/// 4. `write_skill` — the agent's way to author more skills into the same registry
/// 5. `remember` — only when cross-session recall is enabled
///
/// `ask_user` is re-registered with the given sink when one is supplied (the GUI delivers
/// the question through its own input UI; automations auto-answer). `None` keeps the default
/// console-stdin behavior for the CLI.
async fn build_shared_registry(
    cfg: &Config,
    session: &str,
    register_write_skill: bool,
    ask_user: Option<Arc<dyn AskUserSink>>,
) -> Result<Arc<Mutex<ToolRegistry>>> {
    let reg = Arc::new(Mutex::new(register_builtins()));

    {
        let mut r = reg.lock().unwrap();

        if let Some(sink) = ask_user {
            // Override the console `ask_user` from register_builtins with the surface's sink.
            r.register(Arc::new(AskUser::with_sink(sink)));
        }

        if !cfg.mcp_servers.is_empty() {
            // MCP failure is soft — we warn and keep going with the rest of the registry.
            // The CLI/GUI surfaces also surface this through the normal error event channel.
            match connect_mcp_servers(&cfg.mcp_servers).await {
                Ok(mcp) => {
                    logger::info("registry", &format!("MCP tools loaded: {}", mcp.len()));
                    r.extend(&mcp);
                }
                Err(e) => {
                    logger::warn("registry", &format!("MCP connect failed: {e}"));
                }
            }
        }

        // File-based skills (the "generic, agent-authorable" layer).
        let skills = skills::discover_skills(&skills::default_search_roots());
        if !skills.is_empty() {
            logger::info("registry", &format!("skills loaded from disk: {}", skills.len()));
        }
        for s in skills {
            let name = s.name.clone();
            r.register(Arc::new(skills::SkillTool::new(s)));
            logger::info("registry", &format!("skill `{name}` registered"));
        }

        if register_write_skill {
            // Must come last so it can write into the same Arc the engine reads from.
            r.register(Arc::new(WriteSkill::new(Arc::clone(&reg))));
        }

        if cfg.session_recall.unwrap_or(true) {
            if let Some(d) = data_dir().ok() {
                if let Ok(rs) = RecallStore::new(&d) {
                    r.register(Arc::new(Remember::new(rs, session)));
                }
            }
        }
    }

    Ok(reg)
}

fn emit_pretty(ev: &EngineEvent) {
    // Any event other than more reasoning means the thinking run is over, so drop the style
    // before it bleeds into tool logs or the answer.
    if !matches!(ev, EngineEvent::ReasoningDelta { .. }) {
        end_dim();
    }
    match ev {
        EngineEvent::TurnStart { input } => println!("\n>>> {input}"),
        EngineEvent::ReasoningDelta { text } => {
            if SHOW_REASONING.load(Ordering::Relaxed) {
                // Dimmed so thinking never reads as the answer.
                if !DIM_OPEN.swap(true, Ordering::Relaxed) {
                    print!("\x1b[2m");
                }
                print!("{text}");
            }
        }
        EngineEvent::AssistantDelta { text } => {
            end_dim();
            print!("{text}");
        }
        EngineEvent::AssistantMessage {
            text,
            tool_calls,
            reasoning,
            ..
        } => {
            if reasoning.is_some() && SHOW_REASONING.load(Ordering::Relaxed) {
                println!();
            }
            if text.is_some() {
                println!();
            }
            if !tool_calls.is_empty() {
                println!("  ⊙ tools: {}", tool_calls.join(", "));
            }
        }
        EngineEvent::ToolProposed { name, arguments } => println!(
            "  · propose {name} {}",
            serde_json::to_string(arguments).unwrap_or_default()
        ),
        EngineEvent::ToolStarted { name } => println!("  ▶ {name}"),
        EngineEvent::ToolFinished {
            name,
            status,
            result_preview,
        } => println!("  ✔ {name} [{status}] {result_preview}"),
        // `reason` already names the tool, so printing `name` as well just stutters.
        EngineEvent::PermissionRequired { reason, .. } => println!("  ⚠ {reason}"),
        EngineEvent::TurnEnd { status, .. } => println!("  — turn ended: {status}"),
        EngineEvent::Error { error } => println!("  ✗ error: {error}"),
        EngineEvent::Sys(text) => println!("  · {text}"),
        EngineEvent::Interrupted { .. } => println!("  ■ interrupted"),
    }
    let _ = io::stdout().flush();
}


// ---------------------------------------------------------------------------
// GUI (standalone native client — egui/eframe)
// ---------------------------------------------------------------------------

/// Events the engine pushes to the UI thread. Mirrors [`EngineEvent`]; the raw
/// `PermissionRequired` is intentionally dropped (the approver emits a structured
/// `PermissionPrompt` with a request id instead).
#[derive(Serialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
enum GuiEvent {
    TurnStart { input: String },
    ReasoningDelta { text: String },
    AssistantDelta { text: String },
    AssistantMessage { text: Option<String>, tool_calls: Vec<String> },
    ToolProposed { name: String, arguments: Value },
    ToolStarted { name: String },
    ToolFinished { name: String, status: String, result_preview: String },
    PermissionPrompt { request_id: String, name: String, arguments: Value, reason: String },
    /// `ask_user` called by the model: the GUI must collect a free-text answer and post it back
    /// via the ask-answer channel (resolving the pending request id).
    AskUserPrompt { request_id: String, question: String },
    TurnEnd { status: String, iterations: u32 },
    Error { error: String },
    /// A plain status line shown to the user (e.g. "[已停止] …").
    Sys { text: String },
    Done,
}

impl From<EngineEvent> for GuiEvent {
    fn from(ev: EngineEvent) -> Self {
        match ev {
            EngineEvent::TurnStart { input } => GuiEvent::TurnStart { input },
            EngineEvent::ReasoningDelta { text } => GuiEvent::ReasoningDelta { text },
            EngineEvent::AssistantDelta { text } => GuiEvent::AssistantDelta { text },
            EngineEvent::AssistantMessage { text, tool_calls, .. } => {
                GuiEvent::AssistantMessage { text, tool_calls }
            }
            EngineEvent::ToolProposed { name, arguments } => GuiEvent::ToolProposed { name, arguments },
            EngineEvent::ToolStarted { name } => GuiEvent::ToolStarted { name },
            EngineEvent::ToolFinished { name, status, result_preview } => {
                GuiEvent::ToolFinished { name, status, result_preview }
            }
            EngineEvent::PermissionRequired { name, reason, .. } => GuiEvent::PermissionPrompt {
                request_id: String::new(),
                name,
                arguments: Value::Null,
                reason,
            },
            EngineEvent::TurnEnd { status, iterations } => GuiEvent::TurnEnd { status, iterations },
            EngineEvent::Error { error } => GuiEvent::Error { error },
            EngineEvent::Sys(text) => GuiEvent::Sys { text },
            EngineEvent::Interrupted { .. } => GuiEvent::Done,
        }
    }
}

/// One decision posted back from the UI to resolve a pending permission request.
struct DecisionMsg {
    request_id: String,
    decision: String,
}

fn decision_to_outcome(d: &str) -> ApprovalOutcome {
    match d {
        "once" => ApprovalOutcome::Once,
        "always_tool" => ApprovalOutcome::AlwaysTool,
        "always_command" => ApprovalOutcome::AlwaysCommand,
        _ => ApprovalOutcome::Deny,
    }
}

/// Approver that blocks the engine task until the egui UI answers a permission prompt.
struct GuiApprover {
    tx: mpsc::UnboundedSender<GuiEvent>,
    pending: Arc<Mutex<HashMap<String, std::sync::mpsc::Sender<ApprovalOutcome>>>>,
}

impl Approver for GuiApprover {
    fn approve(&self, req: &PermissionRequest) -> ApprovalOutcome {
        let request_id = next_req_id();
        let (otx, orx) = std::sync::mpsc::channel::<ApprovalOutcome>();
        self.pending.lock().unwrap().insert(request_id.clone(), otx);
        // UnboundedSender::send is synchronous — no await / block_on needed.
        let _ = self.tx.send(GuiEvent::PermissionPrompt {
            request_id,
            name: req.tool_name.clone(),
            arguments: req.arguments.clone(),
            reason: req.reason.clone(),
        });
        // Block the engine thread until the egui UI resolves this request. The engine turn
        // runs on its own dedicated OS thread (see GuiApp::send), so this never blocks a
        // tokio worker and cannot deadlock with the decision resolver.
        match orx.recv() {
            Ok(outcome) => outcome,
            Err(_) => ApprovalOutcome::Deny,
        }
    }
}

/// Delivers `ask_user` questions to the GUI's input UI and blocks the engine thread until the
/// answer comes back. Same shape as [`GuiApprover`] — one pending entry per outstanding
/// question, resolved by the egui event loop via the ask-answer channel.
struct GuiAskUserSink {
    tx: mpsc::UnboundedSender<GuiEvent>,
    pending: Arc<Mutex<HashMap<String, std::sync::mpsc::Sender<String>>>>,
}

impl GuiAskUserSink {
    fn new(
        tx: mpsc::UnboundedSender<GuiEvent>,
        pending: Arc<Mutex<HashMap<String, std::sync::mpsc::Sender<String>>>>,
    ) -> Self {
        GuiAskUserSink { tx, pending }
    }
}

impl AskUserSink for GuiAskUserSink {
    fn ask(&self, question: &str) -> String {
        let request_id = next_req_id();
        let (otx, orx) = std::sync::mpsc::channel::<String>();
        self.pending.lock().unwrap().insert(request_id.clone(), otx);
        let _ = self.tx.send(GuiEvent::AskUserPrompt {
            request_id: request_id.clone(),
            question: question.to_string(),
        });
        match orx.recv() {
            Ok(answer) => answer,
            Err(_) => String::new(),
        }
    }
}

static REQ_ID: AtomicU64 = AtomicU64::new(0);
fn next_req_id() -> String {
    format!("r{}", REQ_ID.fetch_add(1, AtomicOrdering::Relaxed))
}

/// A single rendered line in the chat scrollback.
enum DisplayItem {
    User(String),
    Reasoning(String),
    Assistant(String),
    Permission { request_id: String, name: String, args: Value, reason: String },
    AskUser { request_id: String, question: String },
    Sys(String),
}

/// Run one turn against the engine, streaming [`GuiEvent`]s to `event_tx`. Memory is loaded
/// from and saved to the local store, exactly like the CLI.
async fn run_turn_streamed(
    cfg: Arc<Config>,
    pending: Arc<Mutex<HashMap<String, std::sync::mpsc::Sender<ApprovalOutcome>>>>,
    ask_pending: Arc<Mutex<HashMap<String, std::sync::mpsc::Sender<String>>>>,
    event_tx: mpsc::UnboundedSender<GuiEvent>,
    session: String,
    prompt: String,
    model_override: Option<String>,
    mode: String,
    show_reasoning: bool,
    cancel: Arc<AtomicBool>,
    clear_pending: Arc<AtomicBool>,
) {
    let built: Result<(Arc<dyn ProviderClient>, Arc<Mutex<ToolRegistry>>, String, Mode, String)> =
        async {
            let provider = build_provider(&cfg, model_override.as_deref())?;
            let model = resolve_model(&cfg, model_override.as_deref());
            let mode = Mode::from_str(&mode);
            // `ask_user` questions are delivered to the GUI's own input UI, never to a
            // (nonexistent) console — that's what un-sticks multi-turn GUI chats.
            let ask_sink: Arc<dyn AskUserSink> =
                Arc::new(GuiAskUserSink::new(event_tx.clone(), ask_pending.clone()));
            let reg = build_shared_registry(&cfg, &session, true, Some(ask_sink)).await?;
            Ok((provider, reg, model, mode, cfg.effective_instructions()))
        }
        .await;
    let (provider, registry, model, mode, instructions) = match built {
        Ok(v) => v,
        Err(e) => {
            let _ = event_tx.send(GuiEvent::Error { error: e.to_string() });
            let _ = event_tx.send(GuiEvent::Done);
            return;
        }
    };

    if !cfg.mcp_servers.is_empty() {
        match connect_mcp_servers(&cfg.mcp_servers).await {
            Ok(mcp) => {
                logger::info(
                    "engine",
                    &format!(
                        "MCP connected: {} server(s), {} tool(s)",
                        cfg.mcp_servers.len(),
                        mcp.schemas().len()
                    ),
                );
                let count = mcp.schemas().len();
                registry.lock().unwrap().extend(&mcp);
                logger::info("engine", &format!("registered {count} MCP tools"));
            }
            Err(e) => {
                logger::warn("engine", &format!("MCP connect failed: {e}"));
                let _ = event_tx.send(GuiEvent::Error {
                    error: format!("MCP 连接失败: {e}"),
                });
            }
        }
    }

    let approver: Box<dyn Approver> = Box::new(GuiApprover {
        tx: event_tx.clone(),
        pending: pending.clone(),
    });
    let perms = PermissionEngine::new(mode);
    let mut engine = TurnEngine::new_shared(
        provider,
        registry,
        perms,
        model,
        Some(instructions),
        approver,
    );

    engine.set_auto_compress(cfg.auto_compress.unwrap_or(true));
    if let Some(r) = cfg.context_compress_ratio {
        engine.set_auto_compress_ratio(r);
    }
    if let Some(n) = cfg.max_iterations {
        engine.set_max_iterations(n);
    }

    let store = data_dir().ok().and_then(|d| MemoryStore::new(&d).ok());
    if let Some(s) = &store {
        if let Ok(h) = s.load(&session) {
            engine.load_history(h);
        }
    }

    // Log engine progress for post-mortem analysis (see `log_engine_event`), and accumulate
    // streamed text so a user-initiated stop can preserve partial output in history.
    let mut accumulated = String::new();
    let mut stream_started = false;
    let cb = &mut |ev: EngineEvent| {
        match &ev {
            EngineEvent::AssistantDelta { text } => {
                accumulated.push_str(text);
                if !stream_started {
                    stream_started = true;
                    logger::info("engine", "assistant streaming started");
                }
            }
            other => log_engine_event(other),
        }
        if matches!(ev, EngineEvent::PermissionRequired { .. }) {
            return;
        }
        if !show_reasoning && matches!(ev, EngineEvent::ReasoningDelta { .. }) {
            return;
        }
        let _ = event_tx.send(GuiEvent::from(ev));
    };

    let run_fut = engine.run_turn(prompt, cb);

    tokio::select! {
        _ = run_fut => {}
        _ = async {
            loop {
                if cancel.load(Ordering::Relaxed) {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(40)).await;
            }
        } => {
            // User pressed stop. `run_fut` is dropped here, aborting the in-flight model
            // stream / tool call. Preserve what we already have so the chat can be continued.
            if clear_pending.load(Ordering::Relaxed) {
                logger::info("engine", "turn cancelled: END SESSION (history cleared)");
                // "End conversation": wipe this session's history.
                if let Some(s) = &store {
                    let _ = s.save(&session, &[]);
                }
            } else {
                logger::info("engine", "turn cancelled: STOP (partial history preserved)");
                let mut hist = engine.history().to_vec();
                if !accumulated.trim().is_empty() {
                    hist.push(ChatMessage {
                        role: "assistant".to_string(),
                        content: Value::String(accumulated.clone()),
                        tool_call_id: None,
                        tool_calls: Vec::new(),
                    });
                }
                // Never persist a dangling `assistant(tool_calls)` without its tool results:
                // re-sending that on the next turn makes the API reject the whole history
                // with 400. Drop the incomplete tail before saving.
                let hist = sanitize_history(hist);
                if let Some(s) = &store {
                    let _ = s.save(&session, &hist);
                }
                let _ = event_tx.send(GuiEvent::Sys {
                    text: "[已停止] 可继续输入进一步信息，或点「结束对话」清空本会话".into(),
                });
            }
        }
    }

    let _ = event_tx.send(GuiEvent::Done);
}

/// The standalone native GUI client.
struct GuiApp {
    cfg: Arc<Config>,
    event_tx: mpsc::UnboundedSender<GuiEvent>,
    event_rx: mpsc::UnboundedReceiver<GuiEvent>,
    decision_tx: mpsc::UnboundedSender<DecisionMsg>,
    pending: Arc<Mutex<HashMap<String, std::sync::mpsc::Sender<ApprovalOutcome>>>>,
    /// Answers to `ask_user` questions: the GUI posts `(request_id, answer)` back through
    /// `ask_answer_tx`, and `ask_pending` (shared with the engine thread) resolves the wait.
    ask_answer_tx: mpsc::UnboundedSender<(String, String)>,
    ask_pending: Arc<Mutex<HashMap<String, std::sync::mpsc::Sender<String>>>>,
    /// Pending free-text answers, keyed by request id (draft text per outstanding question).
    ask_drafts: HashMap<String, String>,
    sessions: Vec<String>,
    current_session: String,
    model: String,
    mode: String,
    show_reasoning: bool,
    input: String,
    items: Vec<DisplayItem>,
    busy: bool,
    /// Set to `true` to abort the in-flight turn (the "stop" button).
    cancel: Arc<AtomicBool>,
    /// Set to `true` alongside `cancel` to also wipe the session's history ("end conversation").
    clear_pending: Arc<AtomicBool>,
    permission_decided: HashSet<String>,
    /// One-shot request to scroll the chat view to the bottom (used by the "回到底部" button).
    scroll_bottom: bool,
    /// Whether the chat view is currently parked at the bottom (persisted across frames so
    /// new messages only auto-follow when the user hasn't scrolled up to read history).
    at_bottom: bool,
    /// True while a *single* assistant reply is being streamed. Resets at each `AssistantMessage`
    /// (one per engine iteration), so deltas from the next round start a fresh chat bubble
    /// instead of appending to the previous one.
    assistant_stream_open: bool,
    /// Same boundary for the reasoning trace.
    reasoning_stream_open: bool,
    // --- API 配置（左侧栏）---
    /// 服务商协议，默认 "deepseek"（默认 DeepSeek API 协议）。
    api_provider: String,
    /// API Key（界面以密码形式掩码显示）。
    api_key: String,
    /// OpenAI 兼容的 base_url（留空则按服务商取默认值）。
    api_base_url: String,
    /// 模型名（留空则按服务商取默认值）。
    api_model: String,
    /// 测试连接的结果状态文本。
    api_test_status: String,
    /// 测试连接进行中标记，避免重复点击。
    api_testing: bool,
    /// 测试连接结果回传通道。
    api_test_tx: mpsc::UnboundedSender<String>,
    /// 测试连接结果接收端（在 update() 中排空）。
    api_test_rx: mpsc::UnboundedReceiver<String>,
}

impl GuiApp {
    fn new(
        cfg: Arc<Config>,
        event_tx: mpsc::UnboundedSender<GuiEvent>,
        event_rx: mpsc::UnboundedReceiver<GuiEvent>,
        decision_tx: mpsc::UnboundedSender<DecisionMsg>,
        pending: Arc<Mutex<HashMap<String, std::sync::mpsc::Sender<ApprovalOutcome>>>>,
        ask_answer_tx: mpsc::UnboundedSender<(String, String)>,
        ask_pending: Arc<Mutex<HashMap<String, std::sync::mpsc::Sender<String>>>>,
        sessions: Vec<String>,
    ) -> Self {
        let current_session = if sessions.is_empty() {
            "default".to_string()
        } else {
            sessions[0].clone()
        };
        // 左侧栏 API 配置的初始值：服务商默认 deepseek，base_url / model 取该服务商的默认值，
        // api_key 优先沿用已加载配置里的（如本地配置已含 key 则自动填入，界面掩码显示）。
        let api_provider = if cfg.model.provider.is_empty() {
            "deepseek".to_string()
        } else {
            cfg.model.provider.clone()
        };
        let api_base_url = cfg
            .model
            .base_url
            .clone()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| default_base_url(&api_provider));
        let api_model = cfg
            .model
            .model
            .clone()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| default_model(&api_provider));
        let api_key = cfg.model.api_key.clone().unwrap_or_default();
        let (api_test_tx, api_test_rx) = mpsc::unbounded_channel::<String>();
        GuiApp {
            cfg,
            event_tx,
            event_rx,
            decision_tx,
            pending,
            ask_answer_tx,
            ask_pending,
            ask_drafts: HashMap::new(),
            sessions,
            current_session,
            model: String::new(),
            mode: "interactive".to_string(),
            show_reasoning: false,
            input: String::new(),
            items: Vec::new(),
            busy: false,
            cancel: Arc::new(AtomicBool::new(false)),
            clear_pending: Arc::new(AtomicBool::new(false)),
            permission_decided: HashSet::new(),
            scroll_bottom: false,
            at_bottom: true,
            assistant_stream_open: false,
            reasoning_stream_open: false,
            api_provider,
            api_key,
            api_base_url,
            api_model,
            api_test_status: String::new(),
            api_testing: false,
            api_test_tx,
            api_test_rx,
        }
    }

    fn send(&mut self, prompt: String) {
        self.items.push(DisplayItem::User(prompt.clone()));
        self.busy = true;
        logger::info(
            "gui",
            &format!(
                "send session={} model={} mode={} show_reasoning={} prompt_len={} prompt_head={:?}",
                self.current_session,
                self.model,
                self.mode,
                self.show_reasoning,
                prompt.chars().count(),
                prompt.chars().take(60).collect::<String>()
            ),
        );
        // Fresh cancellation tokens for this turn.
        self.cancel = Arc::new(AtomicBool::new(false));
        self.clear_pending = Arc::new(AtomicBool::new(false));
        let cancel = self.cancel.clone();
        let clear_pending = self.clear_pending.clone();
        // 用左侧栏的 API 配置覆盖已加载配置（仅在用户填写时覆盖）。
        let cfg = self.effective_config();
        let pending = self.pending.clone();
        let ask_pending = self.ask_pending.clone();
        let event_tx = self.event_tx.clone();
        let session = self.current_session.clone();
        let model = if self.model.is_empty() {
            None
        } else {
            Some(self.model.clone())
        };
        let mode = self.mode.clone();
        let show_reasoning = self.show_reasoning;
        // Run the engine on a dedicated OS thread with its own tokio runtime so that the
        // synchronous, blocking `GuiApprover::approve` never occupies a GUI-runtime worker
        // (which would deadlock the decision resolver running on the GUI runtime).
        std::thread::spawn(move || {
            let rt = Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
                .expect("failed to build engine runtime");
            rt.block_on(async move {
                run_turn_streamed(
                    cfg,
                    pending,
                    ask_pending,
                    event_tx,
                    session,
                    prompt,
                    model,
                    mode,
                    show_reasoning,
                    cancel,
                    clear_pending,
                )
                .await;
            });
        });
    }

    /// 以左侧栏 API 配置覆盖已加载的 Config，供引擎使用。
    /// 仅当用户填写了对应字段时才覆盖；留空则沿用加载时的配置（含环境变量回退）。
    fn effective_config(&self) -> Arc<Config> {
        let mut cfg = (*self.cfg).clone();
        cfg.model.provider = self.api_provider.clone();
        let key = self.api_key.trim();
        if !key.is_empty() {
            cfg.model.api_key = Some(key.to_string());
        }
        let base = self.api_base_url.trim();
        cfg.model.base_url = if base.is_empty() {
            None
        } else {
            Some(base.to_string())
        };
        let model = self.api_model.trim();
        cfg.model.model = if model.is_empty() {
            None
        } else {
            Some(model.to_string())
        };
        Arc::new(cfg)
    }

    /// 用左侧栏配置发一个最小补全请求，验证 API Key / base_url 是否可用。
    fn test_connection(&mut self) {
        let provider = self.api_provider.clone();
        let key = self.api_key.trim().to_string();
        let base = if self.api_base_url.trim().is_empty() {
            default_base_url(&provider)
        } else {
            self.api_base_url.trim().to_string()
        };
        let model = if self.api_model.trim().is_empty() {
            default_model(&provider)
        } else {
            self.api_model.trim().to_string()
        };
        let tx = self.api_test_tx.clone();
        self.api_testing = true;
        self.api_test_status = "测试中…".to_string();
        std::thread::spawn(move || {
            let rt = Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build();
            let result: anyhow::Result<()> = match rt {
                Ok(rt) => rt.block_on(async move {
                    // ollama 用占位 key；其余先看用户填写的 key，再回退环境变量。
                    let effective_key = if provider == "ollama" {
                        "ollama".to_string()
                    } else if key.is_empty() {
                        std::env::var(match provider.as_str() {
                            "deepseek" => "DEEPSEEK_API_KEY",
                            _ => "OPENAI_API_KEY",
                        })
                        .unwrap_or_default()
                    } else {
                        key.clone()
                    };
                    if provider != "ollama" && effective_key.is_empty() {
                        return Err(anyhow!(
                            "未提供 API Key（请在左侧栏填写，或设置环境变量）"
                        ));
                    }
                    let p = OpenAICompatibleProvider::with_base_url(&effective_key, &base, &model);
                    let req = CompletionRequest {
                        model: model.clone(),
                        messages: vec![ChatMessage::user("ping")],
                        tools: vec![],
                        settings: ModelSettings {
                            max_tokens: Some(5),
                            ..Default::default()
                        },
                    };
                    match tokio::time::timeout(Duration::from_secs(20), p.complete(req)).await {
                        Ok(Ok(_)) => Ok(()),
                        Ok(Err(e)) => Err(e),
                        Err(_) => Err(anyhow!("连接超时（20s）")),
                    }
                }),
                Err(e) => Err(anyhow!("运行时创建失败: {e}")),
            };
            let msg = match result {
                Ok(()) => "✅ 连接成功".to_string(),
                Err(e) => format!("❌ 连接失败: {e}"),
            };
            let _ = tx.send(msg);
        });
    }

    fn decide(&mut self, request_id: String, decision: String) {
        let _ = self.decision_tx.send(DecisionMsg {
            request_id: request_id.clone(),
            decision: decision.clone(),
        });
        self.permission_decided.insert(request_id);
    }

    /// Abort the in-flight turn (the "stop" button). The engine thread watches `cancel`
    /// and drops the streaming future; a `GuiEvent::Done` then re-enables the input.
    fn stop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
        logger::info("gui", "user pressed STOP");
    }

    /// Stop the turn (if running) and clear this session's conversation, so the user can
    /// start a fresh dialogue in the same named session.
    fn end_session(&mut self) {
        let was_busy = self.busy;
        self.cancel.store(true, Ordering::Relaxed);
        logger::info("gui", &format!("user pressed END SESSION (was_busy={was_busy})"));
        if was_busy {
            // Let the engine thread do the clearing so we don't race its save.
            self.clear_pending.store(true, Ordering::Relaxed);
        } else if let Some(store) = data_dir().ok().and_then(|d| MemoryStore::new(&d).ok()) {
            let _ = store.save(&self.current_session, &[]);
        }
        self.items.clear();
    }

    fn switch_session(&mut self, name: String) {
        self.current_session = name.clone();
        if let Some(store) = data_dir().ok().and_then(|d| MemoryStore::new(&d).ok()) {
            if let Ok(msgs) = store.load(&name) {
                self.items = messages_to_items(&msgs, self.show_reasoning);
            } else {
                self.items.clear();
            }
        }
    }

    fn new_session(&mut self) {
        let n = self.sessions.len() + 1;
        let name = format!("session_{n}");
        self.sessions.push(name.clone());
        self.switch_session(name);
    }

    fn apply_event(&mut self, ev: GuiEvent) {
        match ev {
            GuiEvent::ReasoningDelta { text } => {
                if self.reasoning_stream_open {
                    if let Some(DisplayItem::Reasoning(s)) = self.items.last_mut() {
                        s.push_str(&text);
                    } else {
                        self.items.push(DisplayItem::Reasoning(text));
                    }
                } else {
                    // A new reasoning segment begins a fresh trace (don't merge into a
                    // previous round's bubble).
                    self.reasoning_stream_open = true;
                    self.items.push(DisplayItem::Reasoning(text));
                }
            }
            GuiEvent::AssistantDelta { text } => {
                if self.assistant_stream_open {
                    if let Some(DisplayItem::Assistant(s)) = self.items.last_mut() {
                        s.push_str(&text);
                    } else {
                        self.items.push(DisplayItem::Assistant(text));
                    }
                } else {
                    // Fresh assistant reply: start a new bubble for this round.
                    self.assistant_stream_open = true;
                    self.items.push(DisplayItem::Assistant(text));
                }
            }
            GuiEvent::AssistantMessage { .. } => {
                // The engine emits one `AssistantMessage` per iteration *after* the streaming
                // deltas, carrying the full accumulated text. We must NOT append it — the
                // deltas already rendered it, so appending would duplicate every reply.
                // Its real job here is to close the segment: the next `AssistantDelta` opens a
                // new bubble (separate rounds stay separate).
                self.assistant_stream_open = false;
                self.reasoning_stream_open = false;
            }
            GuiEvent::ToolProposed { .. }
            | GuiEvent::ToolStarted { .. }
            | GuiEvent::ToolFinished { .. } => {
                // Debug detail only — see comment on AssistantMessage above. The engine
                // already logs every tool proposal/start/finish with full previews.
            }
            GuiEvent::PermissionPrompt { request_id, name, arguments, reason } => {
                self.items.push(DisplayItem::Permission {
                    request_id,
                    name,
                    args: arguments,
                    reason,
                });
            }
            GuiEvent::AskUserPrompt { request_id, question } => {
                self.items.push(DisplayItem::AskUser {
                    request_id,
                    question,
                });
            }
            GuiEvent::TurnEnd { status, iterations } => {
                // Raw status strings ("max_iterations_exceeded") left users guessing why a turn
                // stopped; spell out the reason and what to do next.
                let label = match status.as_str() {
                    "completed" => format!("— 本轮完成（{iterations} 轮工具调用）"),
                    "max_iterations_exceeded" => format!(
                        "— 本轮因达到工具调用上限而结束（{iterations} 轮）。任务可能未完成，回复「继续」可接着做。"
                    ),
                    "context_limit" => "— 本轮因上下文达到上限而结束。建议开启新会话继续。".to_string(),
                    "error" => "— 本轮因出错而结束。".to_string(),
                    other => format!("— 本轮结束：{other}（{iterations} 轮）"),
                };
                self.items.push(DisplayItem::Sys(label));
            }
            GuiEvent::Error { error } => {
                self.items.push(DisplayItem::Sys(format!("✗ {error}")));
            }
            GuiEvent::Sys { text } => {
                self.items.push(DisplayItem::Sys(text));
            }
            GuiEvent::Done => {
                self.busy = false;
            }
            GuiEvent::TurnStart { .. } => {}
        }
    }
}

impl eframe::App for GuiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        while let Ok(ev) = self.event_rx.try_recv() {
            self.apply_event(ev);
        }

        // 排空「测试连接」结果通道。
        while let Ok(msg) = self.api_test_rx.try_recv() {
            self.api_test_status = msg;
            self.api_testing = false;
        }

        egui::SidePanel::left("sidebar")
            .resizable(true)
            .min_width(180.0)
            .show(ctx, |ui| {
                ui.heading("OpenWorker");
                ui.separator();
                ui.label("会话");
                let cur = self.current_session.clone();
                let sessions: Vec<String> = self.sessions.clone();
                for s in sessions {
                    if ui.selectable_label(s == cur, &s).clicked() {
                        self.switch_session(s);
                    }
                }
                if ui.button("＋ 新建会话").clicked() {
                    self.new_session();
                }
                ui.separator();
                ui.label("模型");
                egui::ComboBox::from_label("")
                    .selected_text(if self.model.is_empty() {
                        "（默认）".to_string()
                    } else {
                        self.model.clone()
                    })
                    .show_ui(ui, |ui| {
                        for m in ["", "deepseek-v4-flash", "deepseek-v4-pro", "gpt-4o-mini", "llama3"] {
                            ui.selectable_value(
                                &mut self.model,
                                m.to_string(),
                                if m.is_empty() { "（默认）" } else { m },
                            );
                        }
                    });
                ui.label("模式");
                egui::ComboBox::from_label("")
                    .selected_text(&self.mode)
                    .show_ui(ui, |ui| {
                        for m in ["interactive", "auto"] {
                            ui.selectable_value(&mut self.mode, m.to_string(), m);
                        }
                    });
                ui.checkbox(&mut self.show_reasoning, "显示思维链");
                ui.separator();
                ui.heading("API 配置");
                ui.label("服务商（默认 DeepSeek）");
                egui::ComboBox::from_label("")
                    .selected_text(&self.api_provider)
                    .show_ui(ui, |ui| {
                        for p in ["deepseek", "openai", "ollama", "custom"] {
                            if ui
                                .selectable_value(&mut self.api_provider, p.to_string(), p)
                                .clicked()
                            {
                                if self.api_base_url.trim().is_empty() {
                                    self.api_base_url = default_base_url(p);
                                }
                                if self.api_model.trim().is_empty() {
                                    self.api_model = default_model(p);
                                }
                            }
                        }
                    });
                ui.label("API Key");
                ui.add(
                    egui::TextEdit::singleline(&mut self.api_key)
                        .password(true)
                        .hint_text("留空则使用环境变量"),
                );
                ui.label("Base URL");
                ui.add(
                    egui::TextEdit::singleline(&mut self.api_base_url)
                        .hint_text(&default_base_url(&self.api_provider)),
                );
                ui.label("模型");
                ui.add(
                    egui::TextEdit::singleline(&mut self.api_model)
                        .hint_text(&default_model(&self.api_provider)),
                );
                if ui
                    .add_enabled(!self.api_testing, egui::Button::new("测试连接"))
                    .clicked()
                {
                    self.test_connection();
                }
                if !self.api_test_status.is_empty() {
                    ui.label(&self.api_test_status);
                }
                ui.separator();
                ui.label(format!("状态: {}", if self.busy { "运行中…" } else { "空闲" }));
            });

        // Input bar lives in a *fixed* bottom panel, NOT inside the CentralPanel after the
        // scroll area. When chat content grows, `ScrollArea` claims every spare pixel of the
        // CentralPanel and would push a sequentially-laid-out input bar off the bottom of the
        // window (the "input box disappeared after a turn" bug). A bottom panel is never
        // displaced by scroll content, so 输入框/发送/停止 stay pinned above the window edge.
        egui::TopBottomPanel::bottom("input_bar")
            .resizable(false)
            .show(ctx, |ui| {
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    let te = egui::TextEdit::multiline(&mut self.input)
                        .hint_text("输入消息，Enter 发送（Shift+Enter 换行）")
                        .desired_rows(2);
                    let resp = ui.add(te);
                    let can_send = !self.input.trim().is_empty() && !self.busy;
                    if self.busy {
                        if ui.button("⏹ 停止").clicked() {
                            self.stop();
                        }
                    } else if ui.button("发送").clicked() && can_send {
                        let prompt = self.input.trim().to_string();
                        self.input.clear();
                        self.send(prompt);
                    }
                    if ui.button("结束对话").clicked() {
                        self.end_session();
                    }
                    if resp.lost_focus()
                        && ui.input(|i| i.key_pressed(egui::Key::Enter))
                        && can_send
                    {
                        let prompt = self.input.trim().to_string();
                        self.input.clear();
                        self.send(prompt);
                    }
                });
                ui.add_space(4.0);
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.push_id("chat_scroll_area", |ui| {
                // Follow new content only while the user is parked at the bottom; once they
                // scroll up to read history, stay put instead of being yanked down.
                let out = egui::ScrollArea::vertical()
                    .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysVisible)
                    .stick_to_bottom(self.scroll_bottom || self.at_bottom)
                    .show(ui, |ui| {
                    let mut action: Option<(String, String)> = None;
                    // Ask-user answers are resolved after the render loop (they go through the
                    // dedicated ask-answer channel, not the permission decision channel).
                    let mut ask_answers: Vec<(String, String)> = Vec::new();
                    for item in &mut self.items {
                        match item {
                            DisplayItem::User(t) => {
                                ui.label(egui::RichText::new(format!("你: {t}")).strong());
                            }
                            DisplayItem::Reasoning(t) => {
                                ui.label(egui::RichText::new(t.clone()).italics().weak());
                            }
                            DisplayItem::Assistant(t) => {
                                ui.label(t.clone());
                            }
                            DisplayItem::Permission { request_id, name, args, reason } => {
                                let rid = request_id.clone();
                                let is_dec = self.permission_decided.contains(&rid);
                                ui.group(|ui| {
                                    ui.label(format!("⚠ 需要授权: {name}"));
                                    ui.label(format!("原因: {reason}"));
                                    ui.collapsing("参数", |ui| {
                                        ui.label(
                                            serde_json::to_string_pretty(args)
                                                .unwrap_or_default(),
                                        )
                                    });
                                    if !is_dec {
                                        ui.horizontal(|ui| {
                                            if ui.button("允许一次").clicked() {
                                                action = Some((rid.clone(), "once".into()));
                                            }
                                            if ui.button("始终允许").clicked() {
                                                action = Some((rid.clone(), "always_tool".into()));
                                            }
                                            if ui.button("拒绝").clicked() {
                                                action = Some((rid.clone(), "deny".into()));
                                            }
                                        });
                                    } else {
                                        ui.label("（已决定）");
                                    }
                                });
                            }
                            DisplayItem::AskUser { request_id, question } => {
                                let rid = request_id.clone();
                                let answered = self.ask_drafts.contains_key(&rid) && self
                                    .permission_decided
                                    .contains(&rid);
                                ui.group(|ui| {
                                    ui.label(format!("❓ {question}"));
                                    let draft =
                                        self.ask_drafts.entry(rid.clone()).or_default();
                                    let resp = ui.add(
                                        egui::TextEdit::singleline(draft)
                                            .hint_text("输入你的回答…"),
                                    );
                                    let submit = ui.button("发送");
                                    let (submit, enter) = (submit.clicked(), resp.lost_focus()
                                        && ui.input(|i| i.key_pressed(egui::Key::Enter)));
                                    if !answered && (submit || enter) {
                                        let answer = draft.clone();
                                        if !answer.trim().is_empty() {
                                            ask_answers.push((rid.clone(), answer));
                                            self.permission_decided.insert(rid.clone());
                                        }
                                    } else if answered {
                                        ui.label("（已回答）");
                                    }
                                });
                            }
                            DisplayItem::Sys(t) => {
                                ui.label(egui::RichText::new(t.clone()).weak());
                            }
                        }
                    }
                    if let Some((rid, dec)) = action {
                        self.decide(rid, dec);
                    }
                    for (rid, answer) in ask_answers {
                        logger::info(
                            "gui",
                            &format!("ask_user answer request_id={rid} len={}", answer.len()),
                        );
                        let _ = self.ask_answer_tx.send((rid, answer));
                    }
                });

            // Parked at the bottom? (used next frame to decide whether to auto-follow new content)
            let at_bottom =
                out.state.offset.y + out.inner_rect.height() >= out.content_size.y - 4.0;
            self.at_bottom = at_bottom;

            // Consume the one-shot scroll request.
            if self.scroll_bottom {
                self.scroll_bottom = false;
            }

            // Floating "back to bottom" button, shown only when the user has scrolled up.
            if !at_bottom {
                egui::Area::new(egui::Id::new("jump_to_bottom"))
                    .anchor(egui::Align2::RIGHT_BOTTOM, egui::vec2(-16.0, -64.0))
                    .show(ctx, |ui| {
                        if ui
                            .button(egui::RichText::new("↓ 回到底部").strong())
                            .clicked()
                        {
                            self.scroll_bottom = true;
                        }
                    });
            }
            });
        });

        ctx.request_repaint();
    }
}

fn messages_to_items(msgs: &[ChatMessage], show_reasoning: bool) -> Vec<DisplayItem> {
    let mut items = Vec::new();
    for m in msgs {
        let v = serde_json::to_value(m).unwrap_or(Value::Null);
        let role = v.get("role").and_then(|r| r.as_str()).unwrap_or("");
        match role {
            "user" => {
                if let Some(c) = v.get("content").and_then(|c| c.as_str()) {
                    if !c.is_empty() {
                        items.push(DisplayItem::User(c.to_string()));
                    }
                }
            }
            "assistant" => {
                if show_reasoning {
                    if let Some(r) = v.get("reasoning").and_then(|r| r.as_str()) {
                        if !r.is_empty() {
                            items.push(DisplayItem::Reasoning(r.to_string()));
                        }
                    }
                }
                if let Some(c) = v.get("content").and_then(|c| c.as_str()) {
                    if !c.is_empty() {
                        items.push(DisplayItem::Assistant(c.to_string()));
                    }
                }
                // Historical `tool_calls` names are skipped (same noise argument as the live
                // path — they live in the log, not the chat).
            }
            _ => {}
        }
    }
    items
}

/// Make egui render CJK (Chinese/Japanese/Korean) text by injecting a system font that
/// actually contains those glyphs. egui's default font is Latin-only, so without this every
/// Chinese label shows up as a missing-glyph box (□). We try the nicest UI fonts first and fall
/// back to whatever CJK-capable system font is present; if none load we silently keep the
/// default (the app still runs, just with tofu for CJK).
/// Translate an engine event into a single structured log line. Called from both the CLI
/// (`run_one`) and the GUI (`run_turn_streamed`) sinks so every path is covered. Per-token
/// `AssistantDelta`s are intentionally not logged (they would flood the file); the aggregate
/// is captured by `AssistantMessage`.
fn log_engine_event(ev: &EngineEvent) {
    match ev {
        EngineEvent::TurnStart { input } => {
            logger::info("engine", &format!("turn start input_len={}", input.chars().count()));
        }
        EngineEvent::AssistantMessage { text, tool_calls, .. } => {
            let n = text.as_ref().map(|t| t.chars().count()).unwrap_or(0);
            logger::info(
                "engine",
                &format!("assistant message chars={n} tool_calls={tool_calls:?}"),
            );
        }
        EngineEvent::ToolProposed { name, arguments } => {
            let a = serde_json::to_string(arguments).unwrap_or_default();
            let a: String = a.chars().take(200).collect();
            logger::info("engine", &format!("tool proposed: {name} args={a:?}"));
        }
        EngineEvent::ToolStarted { name } => {
            logger::info("engine", &format!("tool started: {name}"));
        }
        EngineEvent::ToolFinished { name, status, result_preview } => {
            let p: String = result_preview.chars().take(160).collect();
            logger::info(
                "engine",
                &format!("tool finished: {name} status={status} preview={p:?}"),
            );
        }
        EngineEvent::PermissionRequired { name, reason, .. } => {
            logger::info(
                "engine",
                &format!("permission required: {name} reason={reason}"),
            );
        }
        EngineEvent::Error { error } => {
            logger::error("engine", &format!("error: {error}"));
        }
        EngineEvent::TurnEnd { status, iterations } => {
            logger::info(
                "engine",
                &format!("turn end status={status} iterations={iterations}"),
            );
        }
        _ => {}
    }
}

fn setup_cjk_fonts(ctx: &egui::Context) -> bool {
    let candidates = [
        "C:\\Windows\\Fonts\\msyh.ttc",   // Microsoft YaHei (best looking UI font)
        "C:\\Windows\\Fonts\\simhei.ttf",  // SimHei (always-present single-file TTF)
        "C:\\Windows\\Fonts\\Deng.ttf",    // DengXian
        "C:\\Windows\\Fonts\\msjh.ttc",    // JhengHei
        "C:\\Windows\\Fonts\\simsun.ttc",  // SimSun
        "C:\\Windows\\Fonts\\NotoSansSC-VF.ttf",
    ];
    for path in candidates {
        if let Ok(bytes) = std::fs::read(path) {
            let mut fonts = egui::FontDefinitions::default();
            fonts.font_data.insert(
                "cjk".to_string(),
                egui::FontData::from_owned(bytes),
            );
            // Prepend so CJK glyphs resolve before egui's Latin default font.
            fonts
                .families
                .entry(egui::FontFamily::Proportional)
                .or_default()
                .insert(0, "cjk".to_string());
            fonts
                .families
                .entry(egui::FontFamily::Monospace)
                .or_default()
                .insert(0, "cjk".to_string());
            ctx.set_fonts(fonts);
            return true;
        }
    }
    false
}

async fn cmd_gui(cfg: &Config) -> Result<()> {
    logger::info("gui", "launching native GUI client");
    let rt = Handle::current();
    let cfg = Arc::new(cfg.clone());
    let (event_tx, event_rx) = mpsc::unbounded_channel::<GuiEvent>();
    let (decision_tx, mut decision_rx) = mpsc::unbounded_channel::<DecisionMsg>();
    let pending = Arc::new(Mutex::new(
        HashMap::<String, std::sync::mpsc::Sender<ApprovalOutcome>>::new(),
    ));
    // Ask-user answers: `(request_id, answer)` posted by the egui UI, resolving the pending
    // wait in the engine thread (see `GuiAskUserSink`).
    let (ask_answer_tx, mut ask_answer_rx) = mpsc::unbounded_channel::<(String, String)>();
    let ask_pending = Arc::new(Mutex::new(
        HashMap::<String, std::sync::mpsc::Sender<String>>::new(),
    ));

    // Resolve pending permission requests when the UI posts a decision.
    let pending2 = pending.clone();
    rt.spawn(async move {
        while let Some(d) = decision_rx.recv().await {
            let outcome = decision_to_outcome(&d.decision);
            // Log the resolution of a permission prompt so a "stuck" turn can be diagnosed:
            // if a `permission required` line has no matching `permission decision` line, the
            // UI never answered (modal dismissed / window closed).
            logger::info(
                "gui",
                &format!(
                    "permission decision request_id={} -> {outcome:?}",
                    d.request_id
                ),
            );
            if let Some(tx) = pending2.lock().unwrap().remove(&d.request_id) {
                let _ = tx.send(outcome);
            }
        }
    });

    // Resolve ask_user questions: the UI posts (request_id, answer), the engine thread's
    // `GuiAskUserSink::ask` unblocks with it.
    let ask_pending2 = ask_pending.clone();
    rt.spawn(async move {
        while let Some((rid, answer)) = ask_answer_rx.recv().await {
            if let Some(tx) = ask_pending2.lock().unwrap().remove(&rid) {
                let _ = tx.send(answer);
            }
        }
    });

    let sessions = data_dir()
        .ok()
        .and_then(|d| MemoryStore::new(&d).ok())
        .and_then(|s| s.list_sessions().ok())
        .unwrap_or_default();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size(egui::vec2(1000.0, 700.0))
            .with_min_inner_size(egui::vec2(640.0, 480.0)),
        ..Default::default()
    };

    eframe::run_native(
        "OpenWorker-rs",
        options,
        Box::new(|cc| {
            let cjk = setup_cjk_fonts(&cc.egui_ctx);
            logger::info("gui", &format!("window created, cjk_font_loaded={cjk}"));
            Ok(Box::new(GuiApp::new(
                cfg, event_tx, event_rx, decision_tx, pending, ask_answer_tx, ask_pending,
                sessions,
            )))
        }),
    )
        .map_err(|e| anyhow!("GUI 启动失败: {e}"))?;
    logger::info("gui", "gui closed");
    Ok(())
}
