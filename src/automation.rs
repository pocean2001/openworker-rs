//! Automations — a cron-driven scheduler. Ports the spirit of `coworker/automation/*`.
//!
//! Each automation pairs a natural-language `prompt` with a cron expression. The scheduler
//! computes the next fire time across all automations, sleeps until then, and invokes the
//! [`AutomationRunner`] (provided by the caller — typically a thin wrapper that builds an
//! engine and runs one turn in auto-approve mode).

use std::str::FromStr;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use cron::Schedule;
use serde::Deserialize;

/// A scheduled task definition (mirrors the config schema).
#[derive(Debug, Clone, Deserialize)]
pub struct Automation {
    pub name: String,
    pub prompt: String,
    /// Cron expression. Both the familiar 5-field Unix form (`"0 9 * * 1-5"`, weekdays 09:00)
    /// and the 6/7-field form the `cron` crate expects (`sec min hour dom mon dow [year]`)
    /// are accepted — 5-field expressions get an implicit `0` seconds column.
    pub cron: String,
    #[serde(default)]
    pub session: Option<String>,
}

/// Normalize a 5-field Unix cron expression to the 6-field form the `cron` crate parses.
pub fn normalize_cron(expr: &str) -> String {
    let fields = expr.split_whitespace().count();
    if fields == 5 {
        format!("0 {}", expr.trim())
    } else {
        expr.trim().to_string()
    }
}

/// Parse a (possibly 5-field) cron expression.
pub fn parse_cron(expr: &str) -> Result<Schedule> {
    let normalized = normalize_cron(expr);
    Schedule::from_str(&normalized)
        .map_err(|e| anyhow::anyhow!("invalid cron expression '{}': {}", expr, e))
}

/// Runs a single automation prompt. Implemented by the CLI so the scheduler stays decoupled
/// from the engine internals.
#[async_trait]
pub trait AutomationRunner: Send + Sync {
    async fn run(&self, prompt: &str, session: Option<&str>);
}

/// Cron-driven automation scheduler.
pub struct Scheduler;

impl Scheduler {
    /// Return the next fire time for one automation, or `None` if its cron is unparseable
    /// or has no future occurrence.
    pub fn next_fire(a: &Automation, after: &DateTime<Utc>) -> Option<DateTime<Utc>> {
        match parse_cron(&a.cron) {
            Ok(s) => s.after(after).next(),
            Err(e) => {
                eprintln!("[automation] {}", e);
                None
            }
        }
    }

    /// Drive automations forever: always fire the soonest due task, then recompute.
    pub async fn serve(
        automations: Vec<Automation>,
        runner: Arc<dyn AutomationRunner>,
    ) -> Result<()> {
        if automations.is_empty() {
            println!("[automation] no automations configured; nothing to schedule");
            return Ok(());
        }
        // Validate up-front so typos surface immediately instead of silently never firing.
        for a in &automations {
            parse_cron(&a.cron)?;
        }
        loop {
            let now = Utc::now();
            let mut best: Option<(DateTime<Utc>, &Automation)> = None;
            for a in &automations {
                if let Some(t) = Self::next_fire(a, &now) {
                    if best.map_or(true, |(bt, _)| t < bt) {
                        best = Some((t, a));
                    }
                }
            }
            match best {
                Some((t, a)) => {
                    let dur = (t - now)
                        .to_std()
                        .unwrap_or(std::time::Duration::from_secs(0));
                    println!("[automation] next '{}' at {} (in {:?})", a.name, t, dur);
                    tokio::time::sleep(dur).await;
                    println!("[automation] firing '{}'", a.name);
                    runner.run(&a.prompt, a.session.as_deref()).await;
                }
                None => {
                    println!("[automation] no valid schedules remain; exiting");
                    break;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn five_field_cron_is_normalized() {
        assert_eq!(normalize_cron("0 9 * * 1-5"), "0 0 9 * * 1-5");
        assert_eq!(normalize_cron("0 0 9 * * 1-5"), "0 0 9 * * 1-5");
        assert!(parse_cron("0 9 * * 1-5").is_ok());
    }
}
