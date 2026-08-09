//! Cross-session recall — let a new conversation "remember" what earlier sessions did.
//!
//! [`memory::MemoryStore`](crate::memory::MemoryStore) already persists the raw transcript of each
//! session, but a raw transcript is useless as context: replaying it would blow the budget the
//! auto-compaction in [`engine`](crate::engine) exists to protect. So each session also gets a
//! small, human-readable **recap** at `<data_dir>/recaps/<session>.md`:
//!
//! ```markdown
//! # 会话回顾：session_1
//! _更新于 2026-08-04 05:50_
//!
//! ## 摘要
//! (model-written, regenerated whenever the transcript changes)
//!
//! ## 要点
//! - [2026-08-04 05:50] (appended by the agent's `remember` tool, or by hand)
//! ```
//!
//! Two deliberate choices:
//!
//! * **Markdown, not a database.** The user owns these files and is expected to edit them; a
//!   wrong or stale memory should be fixable in a text editor, not via a migration.
//! * **Summary is derived, notes are authored.** Rewriting a recap regenerates `## 摘要` from the
//!   transcript but *preserves* `## 要点` verbatim — hand-written knowledge is never clobbered by
//!   an automatic refresh.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use anyhow::{Context, Result};

use crate::provider::ChatMessage;

/// Marker prefix of the injected system message. Used to find and replace a previously injected
/// block instead of stacking a new one every turn (histories are reloaded per turn).
pub const RECALL_HEADER: &str = "[跨会话记忆]";

const SUMMARY_HEADING: &str = "## 摘要";
const NOTES_HEADING: &str = "## 要点";

/// Reads and writes per-session recap files.
///
/// `Clone` is just two `PathBuf`s: the engine and the `remember` tool each hold one, and the
/// filesystem is the shared state.
#[derive(Clone)]
pub struct RecallStore {
    /// Where `<session>.jsonl` transcripts live (the same dir `MemoryStore` uses).
    sessions_dir: PathBuf,
    /// Where `<session>.md` recaps live.
    recaps_dir: PathBuf,
}

impl RecallStore {
    pub fn new(data_dir: &Path) -> Result<Self> {
        let recaps_dir = data_dir.join("recaps");
        fs::create_dir_all(&recaps_dir)
            .with_context(|| format!("create recaps dir {}", recaps_dir.display()))?;
        Ok(Self {
            sessions_dir: data_dir.to_path_buf(),
            recaps_dir,
        })
    }

    pub fn recaps_dir(&self) -> &Path {
        &self.recaps_dir
    }

    /// Mirror of `MemoryStore`'s filename sanitisation so the two agree on what a session is.
    fn safe(session: &str) -> String {
        session.replace(['/', '\\', ':'], "_")
    }

    pub fn recap_path(&self, session: &str) -> PathBuf {
        self.recaps_dir.join(format!("{}.md", Self::safe(session)))
    }

    fn transcript_path(&self, session: &str) -> PathBuf {
        self.sessions_dir
            .join(format!("{}.jsonl", Self::safe(session)))
    }

    pub fn read_recap(&self, session: &str) -> Option<String> {
        fs::read_to_string(self.recap_path(session))
            .ok()
            .filter(|s| !s.trim().is_empty())
    }

    /// Load a session's raw transcript (used to build a summary for it).
    pub fn read_transcript(&self, session: &str) -> Vec<ChatMessage> {
        let Ok(txt) = fs::read_to_string(self.transcript_path(session)) else {
            return vec![];
        };
        txt.lines()
            .filter(|l| !l.trim().is_empty())
            .filter_map(|l| serde_json::from_str::<ChatMessage>(l).ok())
            .collect()
    }

    /// True when a session has a transcript but its recap is missing or older than that
    /// transcript — i.e. the summary needs (re)generating.
    pub fn is_stale(&self, session: &str) -> bool {
        let Ok(tm) = fs::metadata(self.transcript_path(session)) else {
            return false;
        };
        if tm.len() == 0 {
            return false;
        }
        let Ok(rm) = fs::metadata(self.recap_path(session)) else {
            return true;
        };
        match (tm.modified(), rm.modified()) {
            (Ok(t), Ok(r)) => t > r,
            _ => false,
        }
    }

    /// Sessions that have a non-empty transcript, most recently touched first, excluding the
    /// session currently being run (its content is already in the live history).
    pub fn recent_sessions(&self, exclude: &str, limit: usize) -> Vec<String> {
        let excluded = Self::safe(exclude);
        let Ok(entries) = fs::read_dir(&self.sessions_dir) else {
            return vec![];
        };
        let mut rows: Vec<(SystemTime, String)> = Vec::new();
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            let Some(stem) = name.strip_suffix(".jsonl") else {
                continue;
            };
            if stem == excluded {
                continue;
            }
            let Ok(md) = entry.metadata() else { continue };
            if md.len() == 0 {
                continue;
            }
            rows.push((
                md.modified().unwrap_or(SystemTime::UNIX_EPOCH),
                stem.to_string(),
            ));
        }
        rows.sort_by(|a, b| b.0.cmp(&a.0));
        rows.into_iter().take(limit).map(|(_, s)| s).collect()
    }

    /// Rewrite a session's recap with a fresh summary, preserving any hand-written notes.
    pub fn write_recap(&self, session: &str, summary: &str) -> Result<PathBuf> {
        let notes = self
            .read_recap(session)
            .map(|t| section_body(&t, NOTES_HEADING))
            .unwrap_or_default();
        let body = format!(
            "# 会话回顾：{session}\n\n\
             <!-- OpenWorker 自动生成。可手工编辑：下次开新会话时本文件会被读入上下文。\n\
             \x20    「摘要」会在该会话有新内容时自动重写，「要点」永远保留、不会被覆盖。 -->\n\
             _更新于 {ts}_\n\n\
             {SUMMARY_HEADING}\n\n{summary}\n\n{NOTES_HEADING}\n\n{notes}\n",
            ts = now_str(),
            summary = summary.trim(),
            notes = notes.trim(),
        );
        let path = self.recap_path(session);
        fs::write(&path, body).with_context(|| format!("write recap {}", path.display()))?;
        Ok(path)
    }

    /// Append a durable note to a session's recap (the `remember` tool).
    ///
    /// Notes go at the end of the file, which `write_recap` keeps as the `## 要点` section, so a
    /// later summary refresh preserves them.
    pub fn append_note(&self, session: &str, note: &str) -> Result<PathBuf> {
        let note = note.trim();
        if note.is_empty() {
            anyhow::bail!("note is empty");
        }
        let mut text = self.read_recap(session).unwrap_or_else(|| {
            format!(
                "# 会话回顾：{session}\n\n\
                 <!-- OpenWorker 自动生成，可手工编辑。 -->\n\n\
                 {SUMMARY_HEADING}\n\n（本会话尚未生成摘要）\n\n{NOTES_HEADING}\n"
            )
        });
        if !text.contains(NOTES_HEADING) {
            if !text.ends_with('\n') {
                text.push('\n');
            }
            text.push_str(&format!("\n{NOTES_HEADING}\n"));
        }
        if !text.ends_with('\n') {
            text.push('\n');
        }
        // Collapse newlines: one note is one bullet, however the model formatted it.
        let flat = note
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        text.push_str(&format!("- [{}] {}\n", now_str(), flat));
        let path = self.recap_path(session);
        fs::write(&path, text).with_context(|| format!("write recap {}", path.display()))?;
        Ok(path)
    }

    /// A model-free recap, used as a fallback when a summarization call is unavailable or fails.
    ///
    /// Writing *something* matters: a recap file that exists stops [`is_stale`](Self::is_stale)
    /// from returning true, so a failing provider can't make every turn retry the same summary.
    pub fn heuristic_recap(messages: &[ChatMessage]) -> String {
        let mut out = String::new();
        let requests: Vec<String> = messages
            .iter()
            .filter(|m| m.role == "user")
            .filter_map(|m| m.content.as_str())
            .map(|s| truncate(s.trim(), 160))
            .filter(|s| !s.is_empty())
            .collect();
        if !requests.is_empty() {
            out.push_str("该会话中用户提出的请求：\n");
            for r in requests.iter().take(6) {
                out.push_str(&format!("- {r}\n"));
            }
            if requests.len() > 6 {
                out.push_str(&format!("- （另有 {} 条更早的请求）\n", requests.len() - 6));
            }
        }
        let touched = touched_paths(messages);
        if !touched.is_empty() {
            out.push_str("\n涉及的文件：\n");
            for p in touched.iter().take(10) {
                out.push_str(&format!("- {p}\n"));
            }
        }
        if let Some(last) = messages
            .iter()
            .rev()
            .find(|m| m.role == "assistant")
            .and_then(|m| m.content.as_str())
            .map(|s| truncate(s.trim(), 400))
            .filter(|s| !s.is_empty())
        {
            out.push_str(&format!("\n最后一次助手回复（节选）：\n{last}\n"));
        }
        if out.trim().is_empty() {
            out.push_str("（该会话没有可提取的内容）");
        }
        out
    }

    /// Assemble the block injected ahead of a turn. `None` when there is nothing worth recalling.
    ///
    /// `max_chars` is a hard ceiling on the whole block; each session additionally gets a fair
    /// share so one enormous recap can't crowd out the others.
    pub fn build_context(&self, sessions: &[String], max_chars: usize) -> Option<String> {
        if sessions.is_empty() || max_chars == 0 {
            return None;
        }
        let per_session = (max_chars / sessions.len()).max(300);
        let mut blocks: Vec<String> = Vec::new();
        let mut used = 0usize;
        for session in sessions {
            let Some(raw) = self.read_recap(session) else {
                continue;
            };
            let body = strip_front_matter(&raw);
            if body.is_empty() {
                continue;
            }
            let remaining = max_chars.saturating_sub(used);
            if remaining < 200 {
                break;
            }
            let piece = truncate(&body, per_session.min(remaining));
            used += piece.chars().count();
            blocks.push(format!("<<< 会话「{session}」>>>\n{piece}"));
        }
        if blocks.is_empty() {
            return None;
        }
        Some(format!(
            "{RECALL_HEADER} 以下是你在此之前几次会话中的工作回顾，按时间从新到旧排列。\n\n\
             {}\n\n\
             使用说明：以上是背景参考，不是当前任务。用户本轮的真实请求以最新的用户消息为准；\
             若回顾内容与当前情况冲突，一律以当前对话为准，不要照搬旧结论。若其中提到的文件路径、\
             结论对本轮有用，请先用工具核实再使用。当你完成实质性工作、或用户告诉你一个需要长期\
             记住的约定时，调用 remember 工具把它记下来。",
            blocks.join("\n\n")
        ))
    }
}

/// Flatten a transcript to plain text for the recap pass, keeping the most recent
/// `max_chars` worth.
///
/// Deliberately *not* replayed as a message list: truncating messages can orphan an
/// `assistant.tool_calls` from its matching `tool` reply, which OpenAI-compatible endpoints
/// reject. Rendering to text sidesteps the pairing rules entirely and costs fewer tokens.
pub fn render_transcript(messages: &[ChatMessage], max_chars: usize) -> String {
    let mut lines: Vec<String> = Vec::new();
    for m in messages {
        let text = m.content.as_str().unwrap_or("").trim();
        match m.role.as_str() {
            // Identical across sessions (and the recall block itself) — pure noise here.
            "system" => continue,
            "user" => {
                if !text.is_empty() {
                    lines.push(format!("【用户】{}", truncate(text, 1_200)));
                }
            }
            "assistant" => {
                if !text.is_empty() {
                    lines.push(format!("【助手】{}", truncate(text, 1_200)));
                }
                for tc in &m.tool_calls {
                    let hint = tc
                        .arguments
                        .get("path")
                        .or_else(|| tc.arguments.get("command"))
                        .and_then(|v| v.as_str())
                        .map(|s| truncate(s, 120))
                        .unwrap_or_default();
                    lines.push(format!("【调用工具】{} {}", tc.name, hint));
                }
            }
            "tool" => {
                if !text.is_empty() {
                    lines.push(format!("【工具结果】{}", truncate(text, 300)));
                }
            }
            other => {
                if !text.is_empty() {
                    lines.push(format!("【{other}】{}", truncate(text, 300)));
                }
            }
        }
    }

    // Keep the tail: the end of a session holds the conclusions and deliverables.
    let mut kept: Vec<&String> = Vec::new();
    let mut used = 0usize;
    let mut truncated = false;
    for line in lines.iter().rev() {
        let n = line.chars().count() + 1;
        if used + n > max_chars {
            truncated = true;
            break;
        }
        used += n;
        kept.push(line);
    }
    kept.reverse();
    let mut out = String::new();
    if truncated {
        out.push_str("（更早的内容已省略）\n");
    }
    out.push_str(
        &kept
            .into_iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
            .join("\n"),
    );
    out
}

/// Collect file paths mentioned in tool-call arguments, for the model-free recap.
fn touched_paths(messages: &[ChatMessage]) -> Vec<String> {
    let mut seen: Vec<String> = Vec::new();
    for m in messages {
        for tc in &m.tool_calls {
            if let Some(p) = tc.arguments.get("path").and_then(|v| v.as_str()) {
                let p = p.to_string();
                if !seen.contains(&p) {
                    seen.push(p);
                }
            }
        }
    }
    seen
}

/// Return the body of a `## heading` section (up to the next `## `), trimmed.
fn section_body(text: &str, heading: &str) -> String {
    let mut out = String::new();
    let mut inside = false;
    for line in text.lines() {
        if line.trim_start().starts_with("## ") {
            if inside {
                break;
            }
            inside = line.trim() == heading;
            continue;
        }
        if inside {
            out.push_str(line);
            out.push('\n');
        }
    }
    out.trim().to_string()
}

/// Drop the `# title`, HTML comment and `_更新于 …_` line so the injected block is pure content.
fn strip_front_matter(text: &str) -> String {
    let mut out = String::new();
    let mut in_comment = false;
    for line in text.lines() {
        let t = line.trim();
        if in_comment {
            if t.contains("-->") {
                in_comment = false;
            }
            continue;
        }
        if t.starts_with("<!--") {
            if !t.contains("-->") {
                in_comment = true;
            }
            continue;
        }
        if t.starts_with("# ") || (t.starts_with("_更新于") && t.ends_with('_')) {
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    out.trim().to_string()
}

fn truncate(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let cut: String = s.chars().take(max_chars.saturating_sub(1)).collect();
    format!("{cut}…")
}

fn now_str() -> String {
    chrono::Local::now().format("%Y-%m-%d %H:%M").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn notes_survive_a_summary_refresh() {
        let dir = std::env::temp_dir().join(format!("ow-recall-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let store = RecallStore::new(&dir).unwrap();
        store.write_recap("s1", "第一版摘要").unwrap();
        store.append_note("s1", "用户偏好 MinGW 工具链").unwrap();
        store.write_recap("s1", "第二版摘要").unwrap();
        let text = store.read_recap("s1").unwrap();
        assert!(text.contains("第二版摘要"));
        assert!(!text.contains("第一版摘要"));
        assert!(text.contains("用户偏好 MinGW 工具链"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn injected_block_drops_front_matter() {
        let dir = std::env::temp_dir().join(format!("ow-recall-ctx-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let store = RecallStore::new(&dir).unwrap();
        store.write_recap("s1", "做了 A、B 两件事").unwrap();
        let ctx = store
            .build_context(&["s1".to_string()], 4000)
            .expect("context");
        assert!(ctx.starts_with(RECALL_HEADER));
        assert!(ctx.contains("做了 A、B 两件事"));
        assert!(!ctx.contains("<!--"));
        assert!(!ctx.contains("# 会话回顾"));
        let _ = fs::remove_dir_all(&dir);
    }
}
