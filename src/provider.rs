//! Provider-agnostic model access layer.
//!
//! Ports `coworker/providers/base.py` + `coworker/providers/openai_provider.py`.
//!
//! The runtime never imports an SDK directly — it talks to a [`ProviderClient`]. The only
//! concrete implementation shipped here, [`OpenAICompatibleProvider`], speaks the OpenAI Chat
//! Completions wire format. Because that format is what the entire OpenAI-compatible world
//! implements, the same code drives OpenAI, DeepSeek, Qwen, GLM, Mistral, *and* a local
//! Ollama server (via `base_url = http://localhost:11434/v1` with a placeholder key).

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

/// A single tool call requested by the model, with parsed (JSON-object) arguments.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// Always a JSON object (matches the OpenAI `function.arguments` shape).
    pub arguments: Value,
}

/// Normalized token counts for one model round-trip.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    pub input: u64,
    pub output: u64,
    pub cache_read: u64,
    pub cache_write: u64,
}

impl TokenUsage {
    /// Prompt-side total — what actually occupied the context window.
    pub fn context_tokens(&self) -> u64 {
        self.input + self.cache_read + self.cache_write
    }
}

/// One assistant response: free text and/or a set of tool calls.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AssistantTurn {
    pub text: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub finish_reason: Option<String>,
    /// Model thinking text (DeepSeek `reasoning_content`, xAI/OpenRouter `reasoning`, ...).
    pub reasoning: Option<String>,
    pub usage: Option<TokenUsage>,
}

/// What a model/provider can do; used for graceful degradation.
#[derive(Debug, Clone, Copy)]
pub struct ModelCapabilities {
    pub tools: bool,
    pub vision: bool,
    pub pdf: bool,
    pub parallel_tool_calls: bool,
    pub streaming: bool,
}

impl Default for ModelCapabilities {
    fn default() -> Self {
        Self {
            tools: true,
            vision: false,
            pdf: false,
            parallel_tool_calls: true,
            streaming: true,
        }
    }
}

/// One streamed piece: a text and/or reasoning delta, and/or (final) the full turn.
#[derive(Debug, Clone, Default)]
pub struct StreamChunk {
    pub text_delta: Option<String>,
    pub reasoning_delta: Option<String>,
    pub turn: Option<AssistantTurn>,
}

/// A chat message in canonical OpenAI shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
}

impl ChatMessage {
    pub fn system(text: &str) -> Self {
        ChatMessage {
            role: "system".into(),
            content: Value::String(text.into()),
            tool_call_id: None,
            tool_calls: vec![],
        }
    }
    pub fn user(text: &str) -> Self {
        ChatMessage {
            role: "user".into(),
            content: Value::String(text.into()),
            tool_call_id: None,
            tool_calls: vec![],
        }
    }
    pub fn assistant(turn: &AssistantTurn) -> Self {
        let content = turn
            .text
            .clone()
            .map(Value::String)
            .unwrap_or(Value::Null);
        ChatMessage {
            role: "assistant".into(),
            content,
            tool_call_id: None,
            tool_calls: turn.tool_calls.clone(),
        }
    }
    pub fn tool(tool_call_id: &str, result: &Value) -> Self {
        let content = match result {
            Value::String(s) => s.clone(),
            other => serde_json::to_string(other).unwrap_or_default(),
        };
        ChatMessage {
            role: "tool".into(),
            content: Value::String(content),
            tool_call_id: Some(tool_call_id.into()),
            tool_calls: vec![],
        }
    }
}

/// An OpenAI-style tool specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpec {
    #[serde(rename = "type")]
    pub r#type: String,
    pub function: FunctionSpec,
}

/// The `function` part of a [`ToolSpec`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionSpec {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

/// Extra, provider-neutral sampling knobs. Everything else is passed through verbatim.
#[derive(Debug, Clone, Default)]
pub struct ModelSettings {
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub extra: HashMap<String, Value>,
}

/// A single completion request.
#[derive(Debug, Clone)]
pub struct CompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub tools: Vec<ToolSpec>,
    pub settings: ModelSettings,
}

/// Provider-agnostic, single-shot completion interface.
///
/// Deliberately blocking-friendly and without its own agent loop — the engine owns the loop.
#[async_trait]
pub trait ProviderClient: Send + Sync {
    /// Return one assistant turn for the given messages/tools.
    async fn complete(&self, req: CompletionRequest) -> Result<AssistantTurn>;

    /// Stream the turn, emitting `text_delta` / `reasoning_delta` chunks as they arrive, and
    /// finishing with a `turn` chunk carrying the full [`AssistantTurn`].
    async fn stream(
        &self,
        req: CompletionRequest,
        emit: &mut (dyn FnMut(StreamChunk) + Send),
    ) -> Result<AssistantTurn>;

    /// Capability flags for a model.
    fn capabilities(&self, model: &str) -> ModelCapabilities;
}

/// An OpenAI Chat Completions client that also serves the OpenAI-compatible world (Ollama,
/// DeepSeek, Qwen, GLM, Mistral, custom endpoints) via `base_url`.
pub struct OpenAICompatibleProvider {
    http: reqwest::Client,
    api_key: Option<String>,
    base_url: String,
    default_model: String,
}

impl OpenAICompatibleProvider {
    /// OpenAI's hosted API.
    pub fn openai(api_key: &str, model: &str) -> Self {
        Self::with_base_url(api_key, "https://api.openai.com/v1", model)
    }

    /// A local Ollama server (`ollama` is the conventional placeholder key).
    pub fn ollama(model: &str) -> Self {
        Self::with_base_url("ollama", "http://localhost:11434/v1", model)
    }

    /// Any OpenAI-shaped endpoint.
    pub fn with_base_url(api_key: &str, base_url: &str, model: &str) -> Self {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(600))
            .build()
            .expect("reqwest client");
        Self {
            http,
            api_key: Some(api_key.to_string()),
            base_url: base_url.trim_end_matches('/').to_string(),
            default_model: model.to_string(),
        }
    }

    fn endpoint(&self) -> String {
        format!("{}/chat/completions", self.base_url)
    }

    fn build_body(&self, req: &CompletionRequest, stream: bool) -> Value {
        let mut m = Map::new();
        // Per-request model wins; fall back to the one this provider was built with.
        let model = if req.model.is_empty() {
            self.default_model.clone()
        } else {
            req.model.clone()
        };
        m.insert("model".into(), Value::String(model));
        m.insert(
            "messages".into(),
            Value::Array(req.messages.iter().map(to_openai_message).collect()),
        );
        if !req.tools.is_empty() {
            m.insert("tools".into(), serde_json::to_value(&req.tools).unwrap());
        }
        if let Some(t) = req.settings.temperature {
            m.insert("temperature".into(), json!(t));
        }
        if let Some(mt) = req.settings.max_tokens {
            m.insert("max_tokens".into(), json!(mt));
        }
        for (k, v) in &req.settings.extra {
            m.insert(k.clone(), v.clone());
        }
        if stream {
            m.insert("stream".into(), Value::Bool(true));
            m.insert("stream_options".into(), json!({ "include_usage": true }));
        }
        Value::Object(m)
    }
}

#[async_trait]
impl ProviderClient for OpenAICompatibleProvider {
    async fn complete(&self, req: CompletionRequest) -> Result<AssistantTurn> {
        let body = self.build_body(&req, false);
        let mut builder = self.http.post(self.endpoint()).json(&body);
        if let Some(key) = &self.api_key {
            builder = builder.bearer_auth(key);
        }
        let resp = builder.send().await?.error_for_status()?;
        let v: Value = resp.json().await?;
        let choice = v
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|a| a.first())
            .ok_or_else(|| anyhow!("no choices in provider response"))?;
        let message = choice
            .get("message")
            .ok_or_else(|| anyhow!("no message in choice"))?;
        let text = message
            .get("content")
            .and_then(|x| x.as_str())
            .map(|s| s.to_string());
        let tool_calls = parse_tool_calls(message.get("tool_calls"));
        let reasoning = message
            .get("reasoning_content")
            .or_else(|| message.get("reasoning"))
            .and_then(|x| x.as_str())
            .map(|s| s.to_string());
        let usage = v.get("usage").and_then(parse_usage);
        let finish_reason = choice
            .get("finish_reason")
            .and_then(|x| x.as_str())
            .map(|s| s.to_string());
        let (text, tool_calls) = salvage(text, tool_calls, &req.tools);
        Ok(AssistantTurn {
            text,
            tool_calls,
            finish_reason,
            reasoning,
            usage,
        })
    }

    async fn stream(
        &self,
        req: CompletionRequest,
        emit: &mut (dyn FnMut(StreamChunk) + Send),
    ) -> Result<AssistantTurn> {
        let body = self.build_body(&req, true);
        let mut builder = self.http.post(self.endpoint()).json(&body);
        if let Some(key) = &self.api_key {
            builder = builder.bearer_auth(key);
        }
        let resp = builder.send().await?.error_for_status()?;

        let mut byte_stream = resp.bytes_stream();
        let mut buf = String::new();
        let mut text_parts = String::new();
        let mut reasoning_parts = String::new();
        let mut tool_accum: Vec<Accum> = Vec::new();
        let mut usage: Option<TokenUsage> = None;
        let mut finish_reason: Option<String> = None;

        use futures::StreamExt;
        while let Some(chunk) = byte_stream.next().await {
            let chunk = chunk?;
            buf.push_str(&String::from_utf8_lossy(&chunk));
            // Process every complete line we have so far.
            while let Some(pos) = buf.find('\n') {
                let mut line: String = buf.drain(..=pos).collect();
                if let Some(stripped) = line.strip_suffix('\r') {
                    line = stripped.to_string();
                }
                let line = line.trim();
                if !line.starts_with("data:") {
                    continue;
                }
                let data = line["data:".len()..].trim();
                if data == "[DONE]" {
                    break;
                }
                if data.is_empty() {
                    continue;
                }
                let v: Value = match serde_json::from_str(data) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                if let Some(u) = v.get("usage").and_then(parse_usage) {
                    usage = Some(u);
                }
                if let Some(choices) = v.get("choices").and_then(|c| c.as_array()) {
                    if let Some(choice) = choices.first() {
                        if let Some(fr) = choice.get("finish_reason").and_then(|x| x.as_str()) {
                            finish_reason = Some(fr.to_string());
                        }
                        if let Some(delta) = choice.get("delta") {
                            if let Some(r) = delta
                                .get("reasoning_content")
                                .or_else(|| delta.get("reasoning"))
                                .and_then(|x| x.as_str())
                            {
                                if !r.is_empty() {
                                    reasoning_parts.push_str(r);
                                    emit(StreamChunk {
                                        reasoning_delta: Some(r.to_string()),
                                        text_delta: None,
                                        turn: None,
                                    });
                                }
                            }
                            if let Some(c) = delta.get("content").and_then(|x| x.as_str()) {
                                if !c.is_empty() {
                                    text_parts.push_str(c);
                                    emit(StreamChunk {
                                        text_delta: Some(c.to_string()),
                                        reasoning_delta: None,
                                        turn: None,
                                    });
                                }
                            }
                            if let Some(tcs) = delta.get("tool_calls").and_then(|x| x.as_array()) {
                                for tc in tcs {
                                    let index = tc
                                        .get("index")
                                        .and_then(|x| x.as_u64())
                                        .unwrap_or(0) as usize;
                                    if index >= tool_accum.len() {
                                        tool_accum.resize_with(index + 1, Accum::default);
                                    }
                                    let acc = &mut tool_accum[index];
                                    if let Some(id) = tc.get("id").and_then(|x| x.as_str()) {
                                        acc.id = id.to_string();
                                    }
                                    if let Some(fn_v) = tc.get("function") {
                                        if let Some(name) =
                                            fn_v.get("name").and_then(|x| x.as_str())
                                        {
                                            acc.name = name.to_string();
                                        }
                                        if let Some(args) =
                                            fn_v.get("arguments").and_then(|x| x.as_str())
                                        {
                                            acc.args.push_str(args);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let mut tool_calls = Vec::new();
        for (i, acc) in tool_accum.into_iter().enumerate() {
            if acc.id.is_empty() && acc.name.is_empty() && acc.args.is_empty() {
                continue;
            }
            let id = if acc.id.is_empty() {
                format!("call_{}", i)
            } else {
                acc.id
            };
            let arguments = parse_json_or_raw(&acc.args);
            tool_calls.push(ToolCall {
                id,
                name: acc.name,
                arguments,
            });
        }
        let text = if text_parts.is_empty() {
            None
        } else {
            Some(text_parts)
        };
        let (text, tool_calls) = salvage(text, tool_calls, &req.tools);
        Ok(AssistantTurn {
            text,
            tool_calls,
            finish_reason,
            reasoning: if reasoning_parts.is_empty() {
                None
            } else {
                Some(reasoning_parts)
            },
            usage,
        })
    }

    fn capabilities(&self, _model: &str) -> ModelCapabilities {
        ModelCapabilities::default()
    }
}

// ---------------------------------------------------------------------------
// Wire helpers
// ---------------------------------------------------------------------------

#[derive(Default, Clone)]
struct Accum {
    id: String,
    name: String,
    args: String,
}

fn parse_tool_calls(v: Option<&Value>) -> Vec<ToolCall> {
    let arr = match v.and_then(|x| x.as_array()) {
        Some(a) => a,
        None => return vec![],
    };
    let mut out = Vec::new();
    for tc in arr {
        let fn_v = match tc.get("function") {
            Some(f) => f,
            None => continue,
        };
        let name = fn_v
            .get("name")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string();
        let raw_args = fn_v.get("arguments").and_then(|x| x.as_str()).unwrap_or("");
        let arguments = if raw_args.is_empty() {
            json!({})
        } else {
            serde_json::from_str(raw_args).unwrap_or_else(|_| json!({ "_raw": raw_args }))
        };
        let id = tc
            .get("id")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string();
        out.push(ToolCall { id, name, arguments });
    }
    out
}

/// Serialize a canonical [`ChatMessage`] into the OpenAI Chat Completions wire shape.
///
/// Assistant `tool_calls` are emitted as `{"id","type":"function","function":{"name","arguments"}}`
/// with `arguments` re-encoded as a JSON *string* — the exact shape OpenAI and its compat peers
/// require, which differs from our internal `ToolCall` (where `arguments` is a parsed object).
fn to_openai_message(m: &ChatMessage) -> Value {
    let mut obj = Map::new();
    obj.insert("role".into(), Value::String(m.role.clone()));

    if !m.tool_calls.is_empty() {
        let tcs: Vec<Value> = m
            .tool_calls
            .iter()
            .map(|tc| {
                let arguments = match &tc.arguments {
                    Value::String(s) => s.clone(),
                    other => serde_json::to_string(other).unwrap_or_else(|_| "{}".to_string()),
                };
                json!({
                    "id": tc.id,
                    "type": "function",
                    "function": { "name": tc.name, "arguments": arguments }
                })
            })
            .collect();
        obj.insert("tool_calls".into(), Value::Array(tcs));
    }

    if m.role == "tool" {
        if let Some(id) = &m.tool_call_id {
            obj.insert("tool_call_id".into(), Value::String(id.clone()));
        }
    }

    match &m.content {
        Value::Null => {
            obj.insert("content".into(), Value::Null);
        }
        other => {
            obj.insert("content".into(), other.clone());
        }
    }
    Value::Object(obj)
}

fn parse_usage(v: &Value) -> Option<TokenUsage> {
    if v.is_null() {
        return None;
    }
    let prompt = v.get("prompt_tokens").and_then(|x| x.as_u64()).unwrap_or(0);
    let details = v.get("prompt_tokens_details");
    let cached = details
        .and_then(|d| d.get("cached_tokens"))
        .and_then(|x| x.as_u64())
        .unwrap_or(0);
    Some(TokenUsage {
        input: prompt.saturating_sub(cached),
        output: v
            .get("completion_tokens")
            .and_then(|x| x.as_u64())
            .unwrap_or(0),
        cache_read: cached,
        cache_write: 0,
    })
}

fn parse_json_or_raw(s: &str) -> Value {
    let s = s.trim();
    match serde_json::from_str::<Value>(s) {
        Ok(v) if v.is_object() => v,
        _ => json!({ "_raw": s }),
    }
}

// ---------------------------------------------------------------------------
// Tool-call salvage — recover calls some models emit as *text* instead of the
// structured `tool_calls` field (notably local Ollama models). Ports the
// `_salvage_tool_calls_from_text` heuristics from openai_provider.py.
// ---------------------------------------------------------------------------

fn salvage(
    text: Option<String>,
    tool_calls: Vec<ToolCall>,
    tools: &[ToolSpec],
) -> (Option<String>, Vec<ToolCall>) {
    if !tool_calls.is_empty() || tools.is_empty() {
        return (text, tool_calls);
    }
    let content = match &text {
        Some(c) if !c.trim().is_empty() => c.clone(),
        _ => return (text, tool_calls),
    };
    let names: Option<HashSet<String>> =
        Some(tools.iter().map(|t| t.function.name.clone()).collect());

    let mut calls: Vec<ToolCall> = Vec::new();

    // 1) <tool_call> … </tool_call> blocks carrying JSON.
    let re_toolcall = Regex::new(r"(?i)<tool_call>\s*").unwrap();
    for m in re_toolcall.find_iter(&content) {
        let j = m.end();
        let b = content.as_bytes();
        if j < b.len() && (b[j] == b'{' || b[j] == b'[') {
            if let Some(sub) = extract_balanced(&content, j) {
                let parsed: Value = serde_json::from_str(&sub).unwrap_or(Value::Null);
                let arr = if parsed.is_array() {
                    parsed.as_array().cloned().unwrap()
                } else {
                    vec![parsed]
                };
                for d in arr {
                    if let Some(c) = call_from_dict(&d, &names) {
                        calls.push(c);
                    }
                }
            }
        }
    }
    if !calls.is_empty() {
        return (None, renumber(calls));
    }

    // 1b) <function=NAME><parameter=K>V</parameter>…</function> (Qwen/Hermes).
    let re_func = Regex::new(r"(?is)<function\s*=\s*([^\s>]+)\s*>(.*?)</function\s*>").unwrap();
    let re_param =
        Regex::new(r"(?is)<parameter\s*=\s*([^\s>]+)\s*>(.*?)</parameter\s*>").unwrap();
    for fm in re_func.captures_iter(&content) {
        let name = fm[1].trim().to_string();
        if let Some(n) = &names {
            if !n.contains(&name) {
                continue;
            }
        }
        let body = &fm[2];
        let mut args = Map::new();
        for pm in re_param.captures_iter(body) {
            let key = pm[1].trim().to_string();
            let val = pm[2].trim().to_string();
            args.insert(key, coerce_param(&val));
        }
        calls.push(ToolCall {
            id: String::new(),
            name,
            arguments: Value::Object(args),
        });
    }
    if !calls.is_empty() {
        return (None, renumber(calls));
    }

    // 2) Embedded {"name": …, "arguments": …} objects, even amid prose.
    for sub in iter_top_objects(&content) {
        if let Ok(d) = serde_json::from_str::<Value>(&sub) {
            if d.is_object() && d.get("name").is_some() {
                if let Some(c) = call_from_dict(&d, &names) {
                    calls.push(c);
                }
            }
        }
    }
    if !calls.is_empty() {
        return (None, renumber(calls));
    }

    // 3) `toolname {args}` / `toolname [args]` shorthand for known tools.
    if let Some(n) = &names {
        for name in n {
            let re =
                Regex::new(&format!(r"{}\s*[:=]?\s*", regex::escape(name))).unwrap();
            if let Some(m) = re.find(&content) {
                let j = m.end();
                let b = content.as_bytes();
                if j < b.len() && (b[j] == b'{' || b[j] == b'[') {
                    if let Some(sub) = extract_balanced(&content, j) {
                        let parsed: Value = serde_json::from_str(&sub).unwrap_or(Value::Null);
                        let args = if parsed.is_object() {
                            parsed
                        } else {
                            json!({})
                        };
                        calls.push(ToolCall {
                            id: String::new(),
                            name: name.clone(),
                            arguments: args,
                        });
                    }
                }
            }
        }
    }
    if !calls.is_empty() {
        return (None, renumber(calls));
    }

    (text, tool_calls)
}

fn coerce_param(raw: &str) -> Value {
    let s = raw.trim();
    if !s.is_empty() && !s.chars().any(|c| c.is_whitespace()) {
        if let Ok(v) = serde_json::from_str::<Value>(s) {
            if v.is_object() || v.is_array() || v.is_number() || v.is_boolean() {
                return v;
            }
        }
    }
    Value::String(s.to_string())
}

fn extract_balanced(s: &str, start: usize) -> Option<String> {
    let bytes = s.as_bytes();
    if start >= bytes.len() {
        return None;
    }
    let open = bytes[start];
    let close = if open == b'[' { b']' } else { b'}' };
    let mut depth: i32 = 0;
    let mut in_str = false;
    let mut esc = false;
    let mut i = start;
    while i < bytes.len() {
        let ch = bytes[i];
        if in_str {
            if esc {
                esc = false;
            } else if ch == b'\\' {
                esc = true;
            } else if ch == b'"' {
                in_str = false;
            }
        } else if ch == b'"' {
            in_str = true;
        } else if ch == open {
            depth += 1;
        } else if ch == close {
            depth -= 1;
            if depth == 0 {
                return Some(s[start..=i].to_string());
            }
        }
        i += 1;
    }
    None
}

fn iter_top_objects(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'{' {
            if let Some(sub) = extract_balanced(s, i) {
                let len = sub.len();
                out.push(sub);
                i += len;
                continue;
            }
        }
        i += 1;
    }
    out
}

fn call_from_dict(d: &Value, names: &Option<HashSet<String>>) -> Option<ToolCall> {
    let name = d.get("name").and_then(|x| x.as_str())?;
    if let Some(n) = names {
        if !n.contains(name) {
            return None;
        }
    }
    let mut args = d
        .get("arguments")
        .cloned()
        .or_else(|| d.get("parameters").cloned())
        .unwrap_or(Value::Null);
    if args.is_null() {
        args = json!({});
    }
    if args.is_string() {
        let s = args.as_str().unwrap().to_string();
        args = serde_json::from_str(&s).unwrap_or(Value::String(s));
    }
    if !args.is_object() {
        args = json!({ "_raw": args });
    }
    Some(ToolCall {
        id: String::new(),
        name: name.to_string(),
        arguments: args,
    })
}

fn renumber(calls: Vec<ToolCall>) -> Vec<ToolCall> {
    calls
        .into_iter()
        .enumerate()
        .map(|(i, mut c)| {
            c.id = format!("call_salvaged_{}", i);
            c
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn salvage_json_object() {
        let tools = vec![ToolSpec {
            r#type: "function".into(),
            function: FunctionSpec {
                name: "run_command".into(),
                description: "".into(),
                parameters: json!({ "type": "object", "properties": { "command": {} } }),
            },
        }];
        let text = Some(r#"Sure: {"name":"run_command","arguments":{"command":"ls"}}"#.to_string());
        let (text, calls) = salvage(text, vec![], &tools);
        assert!(calls.len() == 1);
        assert_eq!(calls[0].name, "run_command");
        assert_eq!(text, None);
    }
}
