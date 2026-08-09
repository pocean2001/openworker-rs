//! Minimal file-based execution logger for OpenWorker-rs.
//!
//! Logs are appended to `%LOCALAPPDATA%\openworker-rs\logs\openworker-YYYY-MM-DD.log`
//! with millisecond timestamps. No external dependency beyond `chrono` + `dirs`, which are
//! already in the dependency tree.
//!
//! The logger is initialized once at startup via `init_logger()`. Until then (or if it fails
//! to open the file) every `log*` call is a silent no-op, so callers never have to thread a
//! logger handle around and logging can never break the application.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use chrono::Local;

static LOG: OnceLock<Mutex<std::fs::File>> = OnceLock::new();

/// Directory where execution logs live: `%LOCALAPPDATA%\openworker-rs\logs`.
pub fn log_dir() -> PathBuf {
    let base = dirs::data_local_dir().unwrap_or_else(|| PathBuf::from("."));
    let dir = base.join("openworker-rs").join("logs");
    let _ = std::fs::create_dir_all(&dir);
    dir
}

/// Open (creating if needed) today's log file. Call exactly once at startup.
/// Failures are swallowed: logging must never break the application.
pub fn init_logger() {
    let today = Local::now().format("%Y-%m-%d").to_string();
    let path = log_dir().join(format!("openworker-{today}.log"));
    match OpenOptions::new().create(true).append(true).open(&path) {
        Ok(f) => {
            let _ = LOG.set(Mutex::new(f));
        }
        Err(e) => {
            eprintln!("[logger] failed to open {path:?}: {e}");
        }
    }
}

#[derive(Clone, Copy)]
pub enum Level {
    Info,
    Warn,
    Error,
}

fn level_str(l: Level) -> &'static str {
    match l {
        Level::Info => "INFO",
        Level::Warn => "WARN",
        Level::Error => "ERROR",
    }
}

/// Write a single log line. No-op until `init_logger()` has succeeded.
pub fn log(level: Level, module: &str, msg: &str) {
    if let Some(m) = LOG.get() {
        let ts = Local::now().format("%Y-%m-%d %H:%M:%S%.3f");
        let line = format!("[{ts}] {} [{module}] {msg}\n", level_str(level));
        if let Ok(mut f) = m.lock() {
            let _ = f.write_all(line.as_bytes());
            let _ = f.flush();
        }
    }
}

#[inline]
pub fn info(module: &str, msg: &str) {
    log(Level::Info, module, msg);
}

#[inline]
pub fn warn(module: &str, msg: &str) {
    log(Level::Warn, module, msg);
}

#[inline]
pub fn error(module: &str, msg: &str) {
    log(Level::Error, module, msg);
}
