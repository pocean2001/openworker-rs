//! The owned agent loop — ports `coworker/engine.py` (TurnEngine).
//!
//! One user turn spans many model⇄tool iterations until the model stops requesting tools, the
//! iteration cap trips, or it's interrupted. Text/reasoning deltas are emitted live; writes and
//! shell calls are gated by the [`PermissionEngine`]; low-risk calls (reads) execute
//! concurrently while everything else runs strictly in order.

use std::sync::{Arc, Mutex};

use anyhow::Result;
use futures::future;
use serde_json::{json, Value};

use crate::permissions::*;
use crate::provider::*;
use crate::recall::{RecallStore, RECALL_HEADER};
use crate::tools::{Tool, ToolRegistry};

/// Events emitted by [`TurnEngine::run_turn`] so surfaces can render progress.
#[derive(Debug, Clone)]
pub enum EngineEvent {
    TurnStart { input: String },
    ReasoningDelta { text: String },
    AssistantDelta { text: String },
    AssistantMessage {
        text: Option<String>,
        tool_calls: Vec<String>,
        reasoning: Option<String>,
        usage: Option<TokenUsage>,
    },
    ToolProposed { name: String, arguments: Value },
    ToolStarted { name: String },
    ToolFinished {
        name: String,
        status: String,
        result_preview: String,
    },
    PermissionRequired {
        name: String,
        arguments: Value,
        reason: String,
    },
    TurnEnd { status: String, iterations: u32 },
    Error { error: String },
    Interrupted { iterations: u32 },
    /// A plain status line surfaced to the user (context-budget warnings, soft stops, …).
    Sys(String),
}

/// Ceiling on tool-call rounds within a single turn. This is a runaway-loop guard, *not* a
/// complexity limit: real tasks ("write a skill, build it, test it") routinely need 15+ rounds, and
/// the context budget + auto-compaction below are what actually keep a long turn affordable.
/// 50 is deliberately generous: the guard's job is to stop an *uncontrolled* loop, while runaway
/// cost is already bounded by the 60k context budget + 60% auto-compact + 80% warning. Too low a
/// ceiling (12, historically) chopped off real multi-stage tasks right before their last step.
const DEFAULT_MAX_ITERATIONS: u32 = 50;

/// Handed to the model for the final tool-less pass after the iteration ceiling trips, so the user
/// gets a real wrap-up instead of the turn silently vanishing mid-task.
const WRAPUP_INSTRUCTION: &str = "已达到本轮工具调用次数上限，不能再调用任何工具。请立刻用简洁的中文向用户\
汇报：①已经完成了什么（含创建/修改的文件路径与验证结果）；②还剩下什么没做；③用户下一步该怎么做（例如回复\
「继续」即可接着完成）。只输出这份小结，不要再尝试调用工具，也不要编造未真正执行过的结果。";

/// Hard safety net: stop a turn before the request would blow the model's context window or
/// burn excessive tokens. The system prompt asks the model to compress proactively, and the engine
/// also auto-compacts older history (see [`TurnEngine::maybe_compress`]) well before this trips.
const DEFAULT_CONTEXT_BUDGET_TOKENS: u32 = 60_000;

/// Fraction of [`DEFAULT_CONTEXT_BUDGET_TOKENS`] at which the engine proactively summarizes older
/// (pre-current-task) history into a single compact message, so a long session keeps running
/// without hitting the hard stop.
const DEFAULT_COMPRESS_THRESHOLD_RATIO: f64 = 0.6;

/// Don't spend a summarization round-trip on trivially small histories.
const MIN_COMPRESS_CHARS: usize = 2_000;

/// How many earlier sessions to recall before a turn.
const DEFAULT_RECALL_SESSIONS: usize = 3;

/// Ceiling on the injected recall block, so remembering the past can't crowd out the present.
const DEFAULT_RECALL_CHARS: usize = 4_000;

/// Cap on summary round-trips spent per turn bringing stale recaps up to date. Without it, the
/// first run after a long break would stall while it summarized every old session at once.
const MAX_RECAPS_PER_TURN: usize = 2;

/// How much of an earlier session's transcript to feed the recap pass (most recent part wins).
const RECAP_SOURCE_CHARS: usize = 24_000;

/// Instruction for the (tool-less) pass that turns a finished session's transcript into its recap.
const RECAP_INSTRUCTION: &str = "请为下面这段已经结束的助手会话写一份简短回顾，供助手在**未来的新会话**中\
快速回忆这次做过什么。必须包含：用户的目标；最终结论或交付物（含创建/修改的文件的准确路径）；重要的决定与\
约束（例如工具链、命名约定、用户明确表达的偏好）；未完成或已知遗留的问题。控制在 300 字以内，用要点列表，\
不要复述对话过程，不要编造对话中没有的内容。只输出正文。";

/// Instruction handed to the model for the (tool-less) summarization pass.
const COMPRESS_INSTRUCTION: &str = "你是 AI 编程助手的对话压缩器。给定一段较早的对话，请输出一段紧凑但信息密集的\
摘要，用于让助手继续当前任务。必须保留：用户的真实目标与明确约束；关键决策及其理由；创建/修改过的文件及准确路径；\
工具调用的结果（成功/失败及重要报错）；尚未完成的事项与下一步。去掉寒暄、重复与冗余细节，不要编造对话中没有的内容。\
只输出摘要正文，不要任何前言。";

/// True for the system message holding the injected cross-session recall block.
fn is_recall_message(m: &ChatMessage) -> bool {
    m.role == "system"
        && m.content
            .as_str()
            .map(|s| s.starts_with(RECALL_HEADER))
            .unwrap_or(false)
}

/// Wiring for cross-session recall: which session is live, where the recaps are, how much of
/// them may be injected. Absent (`None` on the engine) means recall is disabled entirely.
pub struct Recall {
    pub store: RecallStore,
    /// The session being run now — excluded from recall, and the target of `remember`.
    pub session: String,
    /// How many earlier sessions to pull in.
    pub sessions: usize,
    /// Hard ceiling on the injected block, in characters.
    pub max_chars: usize,
}

impl Recall {
    pub fn new(store: RecallStore, session: impl Into<String>) -> Self {
        Recall {
            store,
            session: session.into(),
            sessions: DEFAULT_RECALL_SESSIONS,
            max_chars: DEFAULT_RECALL_CHARS,
        }
    }
}

/// The agent loop.
pub struct TurnEngine {
    provider: Arc<dyn ProviderClient>,
    /// Shared with the surface (and the `write_skill` tool) so that newly authored skills can
    /// be hot-loaded into the live registry without restarting the turn. Tools are themselves
    /// `Arc<dyn Tool>` inside, so cloning the `Arc` is cheap and the lock is held only for the
    /// lookup, never across an `.await`.
    registry: Arc<Mutex<ToolRegistry>>,
    permissions: PermissionEngine,
    model: String,
    max_iterations: u32,
    messages: Vec<ChatMessage>,
    approver: Box<dyn Approver>,
    /// Stop a turn once estimated input tokens exceed this. 0 = no limit.
    context_budget_tokens: u32,
    /// Proactively summarize older history once estimated input tokens exceed
    /// `context_budget_tokens * auto_compress_ratio`. Default on.
    auto_compress: bool,
    /// See [`DEFAULT_COMPRESS_THRESHOLD_RATIO`].
    auto_compress_ratio: f64,
    /// Minimum size (chars) of the compressible region before we bother with a summary call.
    min_compress_chars: usize,
    /// Most recent input token count reported by the provider (accurate when available).
    last_input_tokens: u64,
    /// Whether we've already nudged the user about approaching the budget this turn.
    warned_context: bool,
    /// Whether we've already nudged the model about approaching the iteration ceiling this turn.
    warned_iterations: bool,
    /// Cross-session recall config; `None` disables it.
    recall: Option<Recall>,
}

impl TurnEngine {
    pub fn new(
        provider: Arc<dyn ProviderClient>,
        registry: ToolRegistry,
        permissions: PermissionEngine,
        model: String,
        instructions: Option<String>,
        approver: Box<dyn Approver>,
    ) -> Self {
        let mut engine = TurnEngine {
            provider,
            registry: Arc::new(Mutex::new(registry)),
            permissions,
            model,
            max_iterations: DEFAULT_MAX_ITERATIONS,
            messages: Vec::new(),
            approver,
            context_budget_tokens: DEFAULT_CONTEXT_BUDGET_TOKENS,
            auto_compress: true,
            auto_compress_ratio: DEFAULT_COMPRESS_THRESHOLD_RATIO,
            min_compress_chars: MIN_COMPRESS_CHARS,
            last_input_tokens: 0,
            warned_context: false,
            warned_iterations: false,
            recall: None,
        };
        if let Some(sys) = instructions {
            if !sys.trim().is_empty() {
                engine.messages.push(ChatMessage::system(&sys));
            }
        }
        engine
    }

    /// Build an engine that shares its tool registry with a caller-supplied `Arc<Mutex<…>>`.
    /// Used by the surfaces so they can register `write_skill` (or any other live-mutating
    /// tool) into the same registry the engine dispatches from, *before* the engine runs.
    pub fn new_shared(
        provider: Arc<dyn ProviderClient>,
        registry: Arc<Mutex<ToolRegistry>>,
        permissions: PermissionEngine,
        model: String,
        instructions: Option<String>,
        approver: Box<dyn Approver>,
    ) -> Self {
        let mut engine = TurnEngine {
            provider,
            registry,
            permissions,
            model,
            max_iterations: DEFAULT_MAX_ITERATIONS,
            messages: Vec::new(),
            approver,
            context_budget_tokens: DEFAULT_CONTEXT_BUDGET_TOKENS,
            auto_compress: true,
            auto_compress_ratio: DEFAULT_COMPRESS_THRESHOLD_RATIO,
            min_compress_chars: MIN_COMPRESS_CHARS,
            last_input_tokens: 0,
            warned_context: false,
            warned_iterations: false,
            recall: None,
        };
        if let Some(sys) = instructions {
            if !sys.trim().is_empty() {
                engine.messages.push(ChatMessage::system(&sys));
            }
        }
        engine
    }

    /// Hand back a clone of the live registry's `Arc<Mutex<…>>` so a tool (e.g. `write_skill`)
    /// can register a newly authored skill into the same `ToolRegistry` the engine reads from.
    pub fn registry_handle(&self) -> Arc<Mutex<ToolRegistry>> {
        Arc::clone(&self.registry)
    }

    /// Look up a tool in the shared registry, clone its `Arc`, release the lock, then await.
    /// Holding the lock across an `.await` would stall concurrent dispatches (and the
    /// `write_skill` tool's own re-registration), so we never do that.
    async fn dispatch(&self, name: &str, args: Value) -> Result<Value> {
        let tool = self.registry.lock().unwrap().get(name).cloned();
        match tool {
            Some(t) => t.call(args).await,
            None => Err(anyhow::anyhow!("unknown tool: {name}")),
        }
    }

    pub fn load_history(&mut self, messages: Vec<ChatMessage>) {
        self.messages = sanitize_history(messages);
    }

    pub fn history(&self) -> &[ChatMessage] {
        &self.messages
    }

    /// Enable cross-session recall (see [`Recall`]). Off by default.
    pub fn set_recall(&mut self, recall: Recall) {
        self.recall = Some(recall);
    }

    /// Override the per-turn tool-call ceiling (clamped to at least 1).
    pub fn set_max_iterations(&mut self, n: u32) {
        self.max_iterations = n.max(1);
    }

    /// Override the context-token budget (0 disables the backstop).
    pub fn set_context_budget(&mut self, n: u32) {
        self.context_budget_tokens = n;
    }

    /// Enable/disable proactive history compaction (default on).
    pub fn set_auto_compress(&mut self, on: bool) {
        self.auto_compress = on;
    }

    /// Set the fraction of the context budget at which compaction triggers (clamped to 0.1..0.95).
    pub fn set_auto_compress_ratio(&mut self, ratio: f64) {
        self.auto_compress_ratio = ratio.clamp(0.1, 0.95);
    }

    /// Estimate the input token count of the current message history.
    ///
    /// **Always re-derives from the live messages** rather than trusting the last provider
    /// report verbatim. The provider's `usage` reflects the *previous* request — the tool
    /// results, assistant replies, and model reasoning added since then are invisible to it —
    /// so returning `last_input_tokens` as-is systematically under-counts and lets the context
    /// blow past the 60%-compress threshold straight into the 100% hard stop (the "上下文已达
    /// 上限" truncation you hit mid-task). We instead take the max of:
    ///  1. a conservative chars→tokens conversion of the *current* messages, and
    ///  2. the last real usage (keeps calibration for CJK/heavy-tool contexts where
    ///     chars/4 under-estimates).
    fn estimate_input_tokens(&self) -> u64 {
        let chars: usize = self
            .messages
            .iter()
            .map(|m| serde_json::to_string(&m.content).map(|s| s.chars().count()).unwrap_or(0))
            .sum();
        // ~4 chars/token is a floor, not a ceiling: CJK and tool-heavy payloads cost more.
        let char_est = (chars as u64 + 3) / 4;
        char_est.max(self.last_input_tokens)
    }

    /// Number of leading messages that must survive compaction: the base system prompt plus, when
    /// present, the injected cross-session recall block.
    ///
    /// Identified by content, not by role — the summary [`maybe_compress`](Self::maybe_compress)
    /// produces is *also* a leading system message, and pinning it would let summaries pile up
    /// one per compaction instead of being folded into the next one.
    fn pinned_prefix_len(&self) -> usize {
        let mut n = 0;
        for m in self.messages.iter().take(2) {
            if m.role != "system" {
                break;
            }
            if n == 0 || is_recall_message(m) {
                n += 1;
            } else {
                break;
            }
        }
        n
    }

    /// Load cross-session recall into the context *before* the turn is sent.
    ///
    /// Cheap on the common path: it reads Markdown files. Bringing a *stale* recap up to date
    /// costs one tool-less model call, but that only happens when a session's transcript changed
    /// since its recap was written — so each session is summarized once, not once per turn.
    async fn prepare_recall(&mut self, emit: &mut (dyn FnMut(EngineEvent) + Send)) {
        // Histories are reloaded from disk every turn and the block is persisted with them, so
        // drop any previously injected copy before adding a fresh one.
        self.messages.retain(|m| !is_recall_message(m));

        let Some(recall) = &self.recall else { return };
        let sessions = recall
            .store
            .recent_sessions(&recall.session, recall.sessions);
        if sessions.is_empty() {
            return;
        }
        let stale: Vec<String> = sessions
            .iter()
            .filter(|s| recall.store.is_stale(s))
            .take(MAX_RECAPS_PER_TURN)
            .cloned()
            .collect();

        for session in &stale {
            self.refresh_session_recap(session).await;
        }

        let Some(recall) = &self.recall else { return };
        let Some(block) = recall.store.build_context(&sessions, recall.max_chars) else {
            return;
        };
        let chars = block.chars().count();
        let dir = recall.store.recaps_dir().display().to_string();
        // Insert right after the base system prompt so the recall block precedes all dialogue.
        let at = usize::from(
            self.messages
                .first()
                .map(|m| m.role == "system")
                .unwrap_or(false),
        );
        self.messages.insert(at, ChatMessage::system(&block));
        emit(EngineEvent::Sys(format!(
            "已读取跨会话记忆：{} 个历史会话，约 {} 字符（记忆文件位于 {}，可手工编辑）。",
            sessions.len(),
            chars,
            dir
        )));
    }

    /// (Re)write one earlier session's recap from its transcript.
    ///
    /// The transcript is flattened to plain text rather than replayed as messages: a truncated
    /// message list can leave an `assistant.tool_calls` without its matching `tool` reply, which
    /// most OpenAI-compatible endpoints reject outright.
    ///
    /// Any failure falls back to the model-free recap. Writing *something* is important — the
    /// file's existence is what clears the stale flag, so a failing provider can't make every
    /// subsequent turn retry the same summary.
    async fn refresh_session_recap(&self, session: &str) {
        let Some(recall) = &self.recall else { return };
        let transcript = recall.store.read_transcript(session);
        if transcript.is_empty() {
            return;
        }
        let rendered = crate::recall::render_transcript(&transcript, RECAP_SOURCE_CHARS);
        let summary = if rendered.trim().is_empty() {
            String::new()
        } else {
            let req = CompletionRequest {
                model: self.model.clone(),
                messages: vec![
                    ChatMessage::system(RECAP_INSTRUCTION),
                    ChatMessage::user(&rendered),
                ],
                tools: vec![],
                settings: {
                    let mut s = ModelSettings::default();
                    s.temperature = Some(0.0);
                    s
                },
            };
            match self.provider.complete(req).await {
                Ok(turn) => turn.text.unwrap_or_default().trim().to_string(),
                Err(_) => String::new(),
            }
        };
        let summary = if summary.is_empty() {
            RecallStore::heuristic_recap(&transcript)
        } else {
            summary
        };
        let _ = recall.store.write_recap(session, &summary);
    }

    /// Summarize older history into compact system messages so the running context shrinks
    /// while the active task stays fully intact.
    ///
    /// Two passes, in order:
    ///
    /// 1. **Cross-turn**: everything *before* the current `user` turn (prior, completed work)
    ///    is condensed into a single summary system message. Handles the multi-turn case.
    /// 2. **Intra-turn** (only if there's nothing to compact before the current turn — i.e. a
    ///    single long-running task): fold the *oldest* `assistant(tool_calls) → tool` exchange
    ///    segments *inside* the current turn into one compact system message, newest-first.
    ///    This is what keeps a 30+-tool-call single prompt from hitting the hard context stop:
    ///    without it, every tool result stays verbatim and `idx <= pinned` makes pass 1 a no-op.
    ///
    /// Tool results never outlive their parent assistant call here because tool results always
    /// sit between consecutive `user` messages (or the turn boundary), so folding a whole
    /// exchange segment never leaves a dangling `tool_call_id`.
    async fn maybe_compress(&mut self, emit: &mut (dyn FnMut(EngineEvent) + Send)) {
        let last_user = self.messages.iter().rposition(|m| m.role == "user");
        let Some(idx) = last_user else { return };
        let pinned = self.pinned_prefix_len();

        let target_chars =
            (self.context_budget_tokens as usize * 3 / 4) * 4; // ~budget tokens × 4 chars/token

        // ---- Pass 1: cross-turn history (multi-turn sessions) -------------------------------
        if idx > pinned {
            let region_chars: usize = self.messages[pinned..idx]
                .iter()
                .map(|m| {
                    serde_json::to_string(&m.content)
                        .map(|s| s.chars().count())
                        .unwrap_or(0)
                })
                .sum();
            if region_chars >= self.min_compress_chars {
                emit(EngineEvent::Sys(
                    "上下文接近上限，正在自动压缩早期对话历史…".into(),
                ));
                let summary = self
                    .summarize(&self.messages[pinned..idx])
                    .await;
                if let Some(s) = summary {
                    let summary_msg = ChatMessage {
                        role: "system".to_string(),
                        content: Value::String(format!(
                            "[对话历史摘要 — 已为节省上下文自动压缩此前对话]\n{}",
                            s
                        )),
                        tool_call_id: None,
                        tool_calls: vec![],
                    };
                    let kept_tail: Vec<ChatMessage> = self.messages[idx..].to_vec();
                    let mut new_msgs = Vec::with_capacity(pinned + 1 + kept_tail.len());
                    new_msgs.extend_from_slice(&self.messages[..pinned]);
                    new_msgs.push(summary_msg);
                    new_msgs.extend(kept_tail);
                    self.messages = new_msgs;
                    self.last_input_tokens = 0;
                    self.warned_context = false;
                    emit(EngineEvent::Sys(format!(
                        "已自动压缩早期对话（约 {region_chars} 字符 → 摘要），当前任务不受影响，可继续。",
                    )));
                }
                return;
            }
        }

        // ---- Pass 2: intra-turn compaction (single long-running task) ------------------------
        // There is no meaningful pre-turn history to fold. Instead, fold the oldest tool
        // exchange segments *within* the current turn. Each segment is
        // `assistant(tool_calls) → tool...` following a user message. We leave the first
        // (most recent) exchange intact and collapse the older ones, oldest first.
        let total_chars: usize = self
            .messages
            .iter()
            .map(|m| {
                serde_json::to_string(&m.content)
                    .map(|s| s.chars().count())
                    .unwrap_or(0)
            })
            .sum();
        if total_chars < self.min_compress_chars {
            return;
        }

        // Split the current turn (everything after the pinned prefix) into exchange segments.
        let segments = self.tool_exchange_segments(pinned);
        // Keep the newest segment intact; the rest (oldest first) are foldable.
        if segments.len() < 2 {
            return;
        }
        // Try to fold segments until we're comfortably under the target. Always keep the last
        // (newest) exchange whole so the model still sees the latest tool results.
        let mut new_msgs: Vec<ChatMessage> = Vec::with_capacity(self.messages.len());
        new_msgs.extend_from_slice(&self.messages[..pinned]);
        let mut folded_chars = 0usize;
        let mut folded_segments = 0usize;
        let mut total_out_chars = total_chars;
        for (k, &(s, e, chars)) in segments.iter().enumerate() {
            let is_newest = k + 1 == segments.len();
            if is_newest {
                new_msgs.extend_from_slice(&self.messages[s..e]);
                continue;
            }
            if total_out_chars > target_chars {
                // Fold this old exchange into a one-line note.
                let names: Vec<String> = self.messages[s..e]
                    .iter()
                    .filter(|m| m.role == "assistant")
                    .flat_map(|m| m.tool_calls.iter().map(|tc| tc.name.clone()))
                    .collect();
                let line = format!(
                    "[已压缩第 {} 段工具交换：{}，结果省略]\n",
                    folded_segments + 1,
                    names.join(", ")
                );
                let line_chars = line.chars().count();
                new_msgs.push(ChatMessage {
                    role: "system".to_string(),
                    content: Value::String(line),
                    tool_call_id: None,
                    tool_calls: vec![],
                });
                total_out_chars = total_out_chars.saturating_sub(chars).saturating_add(line_chars);
                folded_chars += chars;
                folded_segments += 1;
            } else {
                new_msgs.extend_from_slice(&self.messages[s..e]);
            }
        }
        if folded_segments == 0 {
            return;
        }
        self.messages = new_msgs;
        self.last_input_tokens = 0;
        self.warned_context = false;
        emit(EngineEvent::Sys(format!(
            "已自动压缩本任务最早的 {folded_segments} 段工具交换（约 {folded_chars} 字符 → 摘要），最近一次工具结果保留，可继续。",
        )));
    }

    /// Split the current turn (everything after `pinned`) into tool-exchange segments.
    /// A segment starts at an `assistant` message carrying tool_calls; the tool results that
    /// answer them belong to it until the next assistant-with-tools, a `user` message (a
    /// natural boundary that must never be folded away), or the turn end.
    /// Returns `(start, end, chars)` per segment, oldest first.
    fn tool_exchange_segments(&self, pinned: usize) -> Vec<(usize, usize, usize)> {
        let mut segments: Vec<(usize, usize, usize)> = Vec::new();
        let mut seg_start: Option<usize> = None;
        let mut seg_chars = 0usize;
        for (i, m) in self.messages[pinned..].iter().enumerate() {
            let abs = pinned + i;
            let is_segment_start = m.role == "assistant" && !m.tool_calls.is_empty();
            if is_segment_start {
                if let Some(s) = seg_start {
                    segments.push((s, abs, seg_chars));
                }
                seg_start = Some(abs);
                seg_chars = serde_json::to_string(&m.content)
                    .map(|s| s.chars().count())
                    .unwrap_or(0);
            } else if m.role == "user" {
                // A user message starts a new exchange; never swallow it into a foldable
                // segment (it carries the task instruction for what follows).
                if let Some(s) = seg_start {
                    segments.push((s, abs, seg_chars));
                    seg_start = None;
                }
            } else if seg_start.is_some() {
                seg_chars += serde_json::to_string(&m.content)
                    .map(|s| s.chars().count())
                    .unwrap_or(0);
            }
        }
        if let Some(s) = seg_start {
            segments.push((s, self.messages.len(), seg_chars));
        }
        segments
    }

    /// One summarization round-trip; `None` means the provider failed or returned nothing.
    async fn summarize(&self, region: &[ChatMessage]) -> Option<String> {
        let mut summ_msgs = Vec::with_capacity(region.len() + 1);
        summ_msgs.push(ChatMessage::system(COMPRESS_INSTRUCTION));
        summ_msgs.extend_from_slice(region);
        let req = CompletionRequest {
            model: self.model.clone(),
            messages: summ_msgs,
            tools: vec![],
            settings: {
                let mut s = ModelSettings::default();
                s.temperature = Some(0.0);
                s
            },
        };
        match self.provider.complete(req).await {
            Ok(turn) => {
                let t = turn.text.unwrap_or_default().trim().to_string();
                if t.is_empty() { None } else { Some(t) }
            }
            Err(_) => None,
        }
    }

    /// One final tool-less pass after the iteration ceiling trips, so the turn ends with a real
    /// hand-off summary (what got done, what's left) instead of stopping mid-task with no reply.
    ///
    /// The wrap-up instruction is only appended to the *request* — it is never persisted into
    /// `self.messages`, so the stored history stays a clean `…tool results → assistant summary`
    /// sequence that the next turn can build on.
    async fn final_wrapup(&mut self, emit: &mut (dyn FnMut(EngineEvent) + Send)) {
        let mut msgs = self.messages.clone();
        msgs.push(ChatMessage::system(WRAPUP_INSTRUCTION));

        let req = CompletionRequest {
            model: self.model.clone(),
            messages: msgs,
            tools: vec![],
            settings: {
                let mut s = ModelSettings::default();
                s.temperature = Some(0.0);
                s
            },
        };

        let turn = match self
            .provider
            .stream(req, &mut |chunk: StreamChunk| {
                if let Some(t) = chunk.text_delta {
                    emit(EngineEvent::AssistantDelta { text: t });
                }
                if let Some(r) = chunk.reasoning_delta {
                    emit(EngineEvent::ReasoningDelta { text: r });
                }
            })
            .await
        {
            Ok(t) => t,
            Err(e) => {
                emit(EngineEvent::Sys(format!(
                    "收尾小结生成失败（不影响已完成的工作）：{e}"
                )));
                return;
            }
        };

        self.messages.push(ChatMessage::assistant(&turn));
        if let Some(u) = &turn.usage {
            self.last_input_tokens = u.input + u.cache_read;
        }
        emit(EngineEvent::AssistantMessage {
            text: turn.text.clone(),
            tool_calls: vec![],
            reasoning: turn.reasoning.clone(),
            usage: turn.usage.clone(),
        });
    }

    /// Run one user turn, emitting events through `emit`.
    pub async fn run_turn(
        &mut self,
        input: String,
        emit: &mut (dyn FnMut(EngineEvent) + Send),
    ) -> Result<()> {
        // Read memory before doing anything else, so the model sees what earlier sessions
        // established before it even reads this turn's request.
        self.prepare_recall(emit).await;

        self.messages.push(ChatMessage::user(&input));
        emit(EngineEvent::TurnStart {
            input: input.clone(),
        });
        self.warned_iterations = false;

        let mut iterations: u32 = 0;
        loop {
            iterations += 1;
            if iterations > self.max_iterations {
                emit(EngineEvent::Sys(format!(
                    "已达单轮工具调用上限（{} 轮），本轮到此结束。下面是进展小结；若任务未完成，直接回复「继续」即可接着做。",
                    self.max_iterations
                )));
                // Spend one final tool-less call so the user gets a real hand-off summary
                // instead of the turn just cutting out mid-task.
                self.final_wrapup(emit).await;
                emit(EngineEvent::TurnEnd {
                    status: "max_iterations_exceeded".into(),
                    iterations,
                });
                return Ok(());
            }
            // Nudge the model to start converging before the ceiling actually trips.
            if !self.warned_iterations
                && self.max_iterations >= 5
                && iterations > self.max_iterations * 4 / 5
            {
                self.warned_iterations = true;
                emit(EngineEvent::Sys(format!(
                    "已用 {}/{} 轮工具调用，接近上限。请优先完成关键步骤并尽快给出结论。",
                    iterations, self.max_iterations
                )));
            }

            // Context-window management.
            if self.context_budget_tokens > 0 {
                let est = self.estimate_input_tokens();
                // Hard backstop: never let the request actually overflow the model.
                if est > self.context_budget_tokens as u64 {
                    emit(EngineEvent::Sys(
                        "上下文已达上限，已停止本轮以避免溢出。建议开启新会话继续，或先让我用一句话总结已完成的进展。".into(),
                    ));
                    emit(EngineEvent::TurnEnd {
                        status: "context_limit".into(),
                        iterations,
                    });
                    return Ok(());
                }
                // Proactive auto-compaction: summarize older (pre-current-task) history so the
                // session survives long exchanges without losing the thread.
                if self.auto_compress
                    && est > (self.context_budget_tokens as f64 * self.auto_compress_ratio) as u64
                {
                    self.maybe_compress(emit).await;
                } else if est > (self.context_budget_tokens as u64 * 4 / 5) && !self.warned_context
                {
                    self.warned_context = true;
                    emit(EngineEvent::Sys(
                        "上下文接近上限（约已用 80%）。请压缩冗余细节，或准备在新会话继续。".into(),
                    ));
                }
            }

            let tools = self.registry.lock().unwrap().schemas();
            let req = CompletionRequest {
                model: self.model.clone(),
                messages: self.messages.clone(),
                tools,
                settings: ModelSettings::default(),
            };

            let turn = match self
                .provider
                .stream(req, &mut |chunk: StreamChunk| {
                    if let Some(t) = chunk.text_delta {
                        emit(EngineEvent::AssistantDelta { text: t });
                    }
                    if let Some(r) = chunk.reasoning_delta {
                        emit(EngineEvent::ReasoningDelta { text: r });
                    }
                })
                .await
            {
                Ok(t) => t,
                Err(e) => {
                    emit(EngineEvent::Error {
                        error: e.to_string(),
                    });
                    emit(EngineEvent::TurnEnd {
                        status: "error".into(),
                        iterations,
                    });
                    return Ok(());
                }
            };

            self.messages.push(ChatMessage::assistant(&turn));
            if let Some(u) = &turn.usage {
                self.last_input_tokens = u.input + u.cache_read;
            }
            emit(EngineEvent::AssistantMessage {
                text: turn.text.clone(),
                tool_calls: turn.tool_calls.iter().map(|c| c.name.clone()).collect(),
                reasoning: turn.reasoning.clone(),
                usage: turn.usage.clone(),
            });

            if turn.tool_calls.is_empty() {
                emit(EngineEvent::TurnEnd {
                    status: "completed".into(),
                    iterations,
                });
                return Ok(());
            }

            self.handle_tool_calls(&turn.tool_calls, emit).await;
        }
    }

    async fn handle_tool_calls(
        &mut self,
        calls: &[ToolCall],
        emit: &mut (dyn FnMut(EngineEvent) + Send),
    ) {
        let mut cleared: Vec<ToolCall> = Vec::new();
        for tc in calls {
            emit(EngineEvent::ToolProposed {
                name: tc.name.clone(),
                arguments: tc.arguments.clone(),
            });
            let meta: Option<ToolMetadata> = self
                .registry
                .lock()
                .unwrap()
                .get(&tc.name)
                .map(|t| t.metadata());
            let decision = self
                .permissions
                .evaluate(&tc.name, &tc.arguments, meta.as_ref());

            if !decision.allowed && decision.needs_user {
                emit(EngineEvent::PermissionRequired {
                    name: tc.name.clone(),
                    arguments: tc.arguments.clone(),
                    reason: decision.reason.clone(),
                });
                let outcome = self.approver.approve(&PermissionRequest {
                    tool_name: tc.name.clone(),
                    arguments: tc.arguments.clone(),
                    metadata: meta.clone(),
                    reason: decision.reason.clone(),
                    tool_call_id: Some(tc.id.clone()),
                });
                match outcome {
                    ApprovalOutcome::Deny => {
                        self.messages
                            .push(ChatMessage::tool(&tc.id, &json!({ "error": "tool call not executed", "reason": "denied by user" })));
                        emit(EngineEvent::ToolFinished {
                            name: tc.name.clone(),
                            status: "denied".into(),
                            result_preview: "denied by user".into(),
                        });
                        continue;
                    }
                    ApprovalOutcome::AlwaysTool => {
                        self.permissions.allow_tool_for_session(&tc.name);
                    }
                    ApprovalOutcome::AlwaysCommand => {
                        let cmd = tc
                            .arguments
                            .get("command")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        self.permissions.allow_command_for_session(&cmd);
                    }
                    ApprovalOutcome::Once => {}
                }
                cleared.push(tc.clone());
            } else if decision.allowed {
                cleared.push(tc.clone());
            } else {
                // Read-only mode, or other hard denial.
                self.messages.push(ChatMessage::tool(
                    &tc.id,
                    &json!({ "error": "tool call not executed", "reason": decision.reason }),
                ));
                emit(EngineEvent::ToolFinished {
                    name: tc.name.clone(),
                    status: "denied".into(),
                    result_preview: decision.reason.clone(),
                });
            }
        }

        // Low-risk calls run concurrently; everything else strictly in order.
        let concurrent: Vec<ToolCall> = if cleared.len() > 1 {
            cleared
                .iter()
                .filter(|tc| self.parallel_safe(tc))
                .cloned()
                .collect()
        } else {
            Vec::new()
        };
        let serial: Vec<ToolCall> = cleared
            .iter()
            .filter(|tc| !concurrent.contains(tc))
            .cloned()
            .collect();

        if !concurrent.is_empty() {
            for tc in &concurrent {
                emit(EngineEvent::ToolStarted { name: tc.name.clone() });
            }
            // Resolve every Arc in a single critical section, then drop the lock so the awaits
            // really run in parallel and the registry stays hot-reloadable mid-turn.
            let resolved: Vec<(String, Option<Arc<dyn Tool>>, Value)> = {
                let r = self.registry.lock().unwrap();
                concurrent
                    .iter()
                    .map(|tc| (tc.name.clone(), r.get(&tc.name).cloned(), tc.arguments.clone()))
                    .collect()
            };
            let results = future::join_all(resolved.into_iter().map(
                |(name, tool, args)| async move {
                    match tool {
                        Some(t) => t.call(args).await,
                        None => Err(anyhow::anyhow!("unknown tool: {name}")),
                    }
                },
            ))
            .await;
            for (tc, res) in concurrent.iter().zip(results) {
                self.record_tool_result(tc, res, emit);
            }
        }

        for tc in &serial {
            emit(EngineEvent::ToolStarted { name: tc.name.clone() });
            let res = self.dispatch(&tc.name, tc.arguments.clone()).await;
            self.record_tool_result(tc, res, emit);
        }
    }

        fn parallel_safe(&self, tc: &ToolCall) -> bool {
            let meta = self
                .registry
                .lock()
                .unwrap()
                .get(&tc.name)
                .map(|t| t.metadata());
        match meta {
            Some(m) => m.risk_level == RiskLevel::Low && !m.requires_approval,
            None => false,
        }
    }

    fn record_tool_result(
        &mut self,
        tc: &ToolCall,
        res: Result<Value>,
        emit: &mut (dyn FnMut(EngineEvent) + Send),
    ) {
        let (result, status) = match res {
            Ok(v) => (v, "ok".to_string()),
            Err(e) => (
                json!({ "error": e.to_string(), "error_type": "tool_error" }),
                "error".to_string(),
            ),
        };
        self.messages.push(ChatMessage::tool(&tc.id, &result));
        let preview = preview(&result);
        emit(EngineEvent::ToolFinished {
            name: tc.name.clone(),
            status,
            result_preview: preview,
        });
    }
}

fn preview(v: &Value) -> String {
    let s = match v {
        Value::String(s) => s.clone(),
        other => serde_json::to_string(other).unwrap_or_default(),
    };
    let s = s.replace('\n', " ");
    if s.chars().count() > 300 {
        let truncated: String = s.chars().take(297).collect();
        format!("{}…", truncated)
    } else {
        s
    }
}

/// Make a message history acceptable to OpenAI/DeepSeek-style chat APIs.
///
/// The wire contract demands strict interleaving: every `assistant` message that declares
/// `tool_calls` must be immediately followed by a `role:"tool"` message for **each** declared
/// id. Interruptions break this invariant: a STOP / approval-dismissed turn can leave history
/// ending with an `assistant` message whose tool never executed, so no `tool` result exists.
/// Sending that raw to the API returns `400 Bad Request` on the *next* turn — exactly the
/// "each new message after a stopped turn fails with 400" symptom.
///
/// The rule: walk forward tracking declared-but-unanswered tool ids. As soon as a new
/// `assistant` message (tool or plain text) shows up while answers are still outstanding, the
/// tail is corrupt — drop it. At the end, if any tool id is still unanswered, strip the
/// incomplete tool-call segment entirely.
pub fn sanitize_history(messages: Vec<ChatMessage>) -> Vec<ChatMessage> {
    let mut out: Vec<ChatMessage> = Vec::with_capacity(messages.len());
    let mut pending: std::collections::HashSet<String> = std::collections::HashSet::new();

    for m in messages {
        if m.role == "assistant" && !m.tool_calls.is_empty() {
            if !pending.is_empty() {
                // A new assistant turn starts while earlier tool results are still owed:
                // the following tail is unrecoverable garbage for the API. Stop here.
                break;
            }
            for tc in &m.tool_calls {
                pending.insert(tc.id.clone());
            }
            out.push(m);
        } else if m.role == "tool" {
            match &m.tool_call_id {
                Some(id) if pending.remove(id) => out.push(m),
                // Unknown / orphan tool result (no matching declaration): drop it.
                _ => {}
            }
        } else {
            // system / user / plain-text assistant.
            if !pending.is_empty() {
                // A plain-text assistant (or user) message appears while tool results are
                // still owed — the middle of the exchange is broken; drop the tail.
                break;
            }
            out.push(m);
        }
    }

    // History ended with an unanswered tool declaration (e.g. stop mid-approval): strip the
    // dangling `assistant(tool_calls)` plus any partial results after it.
    if !pending.is_empty() {
        while let Some(last) = out.last() {
            if last.role == "assistant" && !last.tool_calls.is_empty() {
                out.pop();
                break;
            }
            out.pop();
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tool_msg(id: &str) -> ChatMessage {
        ChatMessage {
            role: "tool".to_string(),
            content: Value::String("ok".into()),
            tool_call_id: Some(id.to_string()),
            tool_calls: vec![],
        }
    }

    fn assistant_with_tools(ids: &[&str]) -> ChatMessage {
        ChatMessage {
            role: "assistant".to_string(),
            content: Value::String("".into()),
            tool_call_id: None,
            tool_calls: ids
                .iter()
                .map(|id| ToolCall {
                    id: id.to_string(),
                    name: "x".into(),
                    arguments: json!({}),
                })
                .collect(),
        }
    }

    fn plain_assistant(text: &str) -> ChatMessage {
        ChatMessage {
            role: "assistant".to_string(),
            content: Value::String(text.into()),
            tool_call_id: None,
            tool_calls: vec![],
        }
    }

    fn user(text: &str) -> ChatMessage {
        ChatMessage {
            role: "user".to_string(),
            content: Value::String(text.into()),
            tool_call_id: None,
            tool_calls: vec![],
        }
    }

    #[test]
    fn complete_history_passes_through_untouched() {
        let hist = vec![
            user("hi"),
            assistant_with_tools(&["a", "b"]),
            tool_msg("a"),
            tool_msg("b"),
            plain_assistant("done"),
        ];
        let out = sanitize_history(hist.clone());
        assert_eq!(out.len(), hist.len());
    }

    #[test]
    fn dangling_tool_call_tail_is_stripped() {
        // The exact failure mode: assistant declared a tool, approval/STOP interrupted before
        // the tool result was recorded. Re-sending this to DeepSeek returns 400.
        let hist = vec![user("hi"), assistant_with_tools(&["w1"])];
        let out = sanitize_history(hist);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].role, "user");
    }

    #[test]
    fn partial_tool_results_are_stripped_with_the_tail() {
        // assistant declared a+b, only a got answered, then interruption. Both the dangling
        // declaration and the orphan-ish partial must go.
        let hist = vec![
            user("hi"),
            assistant_with_tools(&["a", "b"]),
            tool_msg("a"),
        ];
        let out = sanitize_history(hist);
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn mid_exchange_user_message_breaks_and_truncates() {
        // Corrupt ordering that can't happen with the engine today but the API would reject.
        let hist = vec![
            user("hi"),
            assistant_with_tools(&["a"]),
            user("wait"),
            tool_msg("a"),
            plain_assistant("done"),
        ];
        let out = sanitize_history(hist);
        // user "wait" arrives while tool "a" is still owed → cut there. The dangling
        // `assistant(tool_calls=["a"])` that preceded it is also stripped (no tool result),
        // leaving just the intact opening user turn.
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].role, "user");
    }

    #[test]
    fn orphan_tool_result_is_dropped() {
        let hist = vec![user("hi"), tool_msg("ghost")];
        let out = sanitize_history(hist);
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn tool_exchange_segments_group_by_assistant_calls() {
        // Single-turn long task: user → 3 tool exchanges (assistant+tool pairs).
        let mut eng = test_engine();
        eng.messages = vec![
            ChatMessage::system("sys"),
            user("long task"),
            assistant_with_tools(&["t1"]),
            tool_msg("t1"),
            assistant_with_tools(&["t2"]),
            tool_msg("t2"),
            assistant_with_tools(&["t3"]),
            tool_msg("t3"),
        ];
        let pinned = 1; // just the system message
        let segs = eng.tool_exchange_segments(pinned);
        assert_eq!(segs.len(), 3, "got {segs:?}");
        // Oldest first: (start, end, chars). Each segment ends where the next assistant
        // tool-call starts; the last one runs to the end of the messages.
        assert_eq!((segs[0].0, segs[0].1), (2, 4), "got {segs:?}");
        assert_eq!((segs[1].0, segs[1].1), (4, 6), "got {segs:?}");
        assert_eq!((segs[2].0, segs[2].1), (6, 8), "got {segs:?}");
        // Chars are cumulative per segment (assistant + its tool results).
        assert!(segs[2].2 >= 4, "got {segs:?}");
    }

    #[test]
    fn folded_segments_leave_no_dangling_tool_calls() {
        // After folding the two oldest exchanges, sanitize_history must still be clean
        // (no assistant tool_calls without results), because we keep the newest segment whole.
        let mut eng = test_engine();
        eng.messages = vec![
            ChatMessage::system("sys"),
            user("long task"),
            assistant_with_tools(&["t1"]),
            tool_msg("t1"),
            assistant_with_tools(&["t2"]),
            tool_msg("t2"),
            assistant_with_tools(&["t3"]),
            tool_msg("t3"),
        ];
        // Simulate folding segments [0..3) and [3..5) into summary system notes, keeping [5..7).
        let folded: Vec<ChatMessage> = vec![
            ChatMessage::system("sys"),
            user("long task"),
            ChatMessage {
                role: "system".to_string(),
                content: Value::String("[已压缩工具交换…]".into()),
                tool_call_id: None,
                tool_calls: vec![],
            },
            assistant_with_tools(&["t3"]),
            tool_msg("t3"),
        ];
        let expected_len = folded.len();
        let clean = sanitize_history(folded);
        assert_eq!(clean.len(), expected_len, "folding must not create dangling tool calls");
    }

    #[test]
    fn mid_turn_user_message_creates_extra_boundary() {
        let mut eng = test_engine();
        eng.messages = vec![
            ChatMessage::system("sys"),
            user("task A"),
            assistant_with_tools(&["a1"]),
            tool_msg("a1"),
            user("task B"),
            assistant_with_tools(&["b1"]),
            tool_msg("b1"),
        ];
        let segs = eng.tool_exchange_segments(1);
        // Two exchanges, each in its own user-scoped turn; the user message (index 4)
        // terminates segment 1 and must never be folded into it.
        assert_eq!(segs.len(), 2);
        assert_eq!((segs[0].0, segs[0].1), (2, 4), "got {segs:?}");
        assert_eq!((segs[1].0, segs[1].1), (5, 7), "got {segs:?}");
    }

    #[test]
    fn user_messages_are_never_included_in_foldable_segments() {
        // Regression guard: folding must never delete a user instruction. Every segment's
        // span must contain zero `user` messages.
        let mut eng = test_engine();
        eng.messages = vec![
            ChatMessage::system("sys"),
            user("task A"),
            assistant_with_tools(&["a1"]),
            tool_msg("a1"),
            user("task B"),
            assistant_with_tools(&["b1"]),
            tool_msg("b1"),
            assistant_with_tools(&["c1"]),
            tool_msg("c1"),
        ];
        let pinned = 1;
        let segs = eng.tool_exchange_segments(pinned);
        assert!(segs.len() >= 2);
        for (s, e, _) in &segs {
            assert!(
                !eng.messages[*s..*e].iter().any(|m| m.role == "user"),
                "segment {s}..{e} contains a user message"
            );
        }
    }

    fn test_engine() -> TurnEngine {
        TurnEngine::new(
            Arc::new(crate::provider::OpenAICompatibleProvider::ollama("x")),
            crate::tools::register_builtins(),
            crate::permissions::PermissionEngine::new(crate::permissions::Mode::Auto),
            "test-model".to_string(),
            None,
            Box::new(crate::permissions::AutoApprover),
        )
    }
}
