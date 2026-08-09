//! Conversation memory — a dependency-free JSONL store.
//!
//! The upstream uses SQLite; we deliberately store one JSON object per message line so the
//! crate has zero C-dependencies and the history is trivially inspectable/editable by the user.
//! (Swap in SQLite later without changing the `ChatMessage` contract.)

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::provider::ChatMessage;

/// A file-backed conversation store. One `<session>.jsonl` file per session.
pub struct MemoryStore {
    dir: PathBuf,
}

impl MemoryStore {
    pub fn new(dir: &Path) -> Result<Self> {
        fs::create_dir_all(dir)
            .with_context(|| format!("create memory dir {}", dir.display()))?;
        Ok(Self { dir: dir.to_path_buf() })
    }

    fn path(&self, session: &str) -> PathBuf {
        let safe = session.replace(['/', '\\', ':'], "_");
        self.dir.join(format!("{}.jsonl", safe))
    }

    /// Load a session's message history (empty vec if it doesn't exist yet).
    pub fn load(&self, session: &str) -> Result<Vec<ChatMessage>> {
        let p = self.path(session);
        if !p.exists() {
            return Ok(vec![]);
        }
        let txt = fs::read_to_string(&p)?;
        let mut out = Vec::new();
        for line in txt.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Ok(m) = serde_json::from_str::<ChatMessage>(line) {
                out.push(m);
            }
        }
        Ok(out)
    }

    /// Overwrite a session's history with the given messages.
    pub fn save(&self, session: &str, messages: &[ChatMessage]) -> Result<()> {
        let mut s = String::new();
        for m in messages {
            s.push_str(&serde_json::to_string(m)?);
            s.push('\n');
        }
        fs::write(self.path(session), s)?;
        Ok(())
    }

    /// Append a single message (cheap for streaming-style history growth).
    pub fn append(&self, session: &str, message: &ChatMessage) -> Result<()> {
        let mut f = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.path(session))?;
        f.write_all(serde_json::to_string(message)?.as_bytes())?;
        f.write_all(b"\n")?;
        Ok(())
    }

    /// List available session ids.
    pub fn list_sessions(&self) -> Result<Vec<String>> {
        let mut out = Vec::new();
        for entry in fs::read_dir(&self.dir)? {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().to_string();
            if let Some(stem) = name.strip_suffix(".jsonl") {
                out.push(stem.to_string());
            }
        }
        out.sort();
        Ok(out)
    }
}
