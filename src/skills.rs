//! Generic, file-based skills.
//!
//! A **skill** is a self-contained capability the agent can call, packaged as a directory:
//!
//! ```text
//! .openworker/skills/<name>/SKILL.md
//! .openworker/skills/<name>/main.py     # or whatever the manifest points at
//! ```
//!
//! `SKILL.md` is Markdown with a YAML frontmatter that names the skill, declares its JSON-Schema
//! parameters, the risk level the permission engine should charge, and the script entry point.
//! The body of the file is a longer human-readable description that becomes part of the model
//! prompt so it knows when to call the skill.
//!
//! The script receives the tool-call arguments as JSON on **stdin** and is expected to print
//! either a JSON value (object/array/scalar) to **stdout**, or plain text which we wrap into
//! `{"output": "..."}`. stderr is captured into the result under `_stderr` for debugging and
//! never causes failure on its own.
//!
//! Discovery scans two roots in this order, last-wins so a project can shadow a user skill:
//!
//! 1. `~/.openworker/skills/` — user-level (cross-project)
//! 2. `<cwd>/.openworker/skills/` — project-level (ship in-repo, share with the team)
//!
//! A `SKILL.md` whose `name:` field starts with `_` is treated as a template / docs and is
//! **not** registered. Anything with a reserved built-in name (`get_weather`, `read_file`, …) is
//! refused at load time so user skills can never silently shadow a built-in.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use async_trait::async_trait;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

use crate::permissions::{RiskLevel, ToolMetadata};
use crate::provider::{FunctionSpec, ToolSpec};
use crate::tools::Tool;

/// A name must match this so the tool registry, JSON-RPC over MCP, and the permission resolver
/// can all rely on a stable ASCII identifier. (MCP enforces a tighter subset; this is the union.)
fn skill_name_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^[a-z][a-z0-9_]{0,62}[a-z0-9]$").unwrap())
}

/// Reserved names that user/project skills must never replace.
pub const RESERVED_SKILL_NAMES: &[&str] = &[
    "read_file", "write_file", "list_dir", "run_command", "web_fetch", "ask_user",
    "remember", "write_skill",
];

/// A parsed, validated SKILL.md.
#[derive(Debug, Clone)]
pub struct Skill {
    pub name: String,
    pub description: String,
    pub parameters: Value,
    pub risk: RiskLevel,
    pub runtime: String,
    pub entry: PathBuf,
    pub dir: PathBuf,
    pub timeout_secs: u32,
}

/// Raw YAML frontmatter shape. We keep it loose on `parameters` so users can hand-author
/// reasonable JSON-Schema fragments without us round-tripping every keyword.
#[derive(Debug, Deserialize, Serialize)]
struct RawManifest {
    name: String,
    description: String,
    #[serde(default)]
    parameters: Option<Value>,
    #[serde(default = "default_risk")]
    risk: String,
    #[serde(default = "default_runtime")]
    runtime: String,
    #[serde(default)]
    entry: Option<String>,
    #[serde(default = "default_timeout")]
    timeout_secs: u32,
}

fn default_risk() -> String {
    "medium".into()
}
fn default_runtime() -> String {
    "auto".into()
}
fn default_timeout() -> u32 {
    30
}

/// Split a SKILL.md into `(yaml, body)`. Frontmatter is delimited by `---` lines, as is
/// conventional. If no frontmatter is present, `yaml` is empty and the whole file is body.
pub fn split_frontmatter(text: &str) -> (String, String) {
    let trimmed = text.trim_start_matches('\u{feff}');
    if !trimmed.starts_with("---") {
        return (String::new(), trimmed.to_string());
    }
    let after_first = &trimmed[3..];
    let after_first = after_first.trim_start_matches(|c| c == '\r' || c == '\n');
    if let Some(end_rel) = after_first.find("\n---") {
        let yaml = after_first[..end_rel].to_string();
        let after_marker = &after_first[end_rel + 4..];
        let body = after_marker
            .trim_start_matches(|c| c == '\r' || c == '\n')
            .to_string();
        (yaml, body)
    } else {
        (String::new(), trimmed.to_string())
    }
}

/// Parse + validate a SKILL.md. `dir` is the directory the file lives in, so the entry script
/// is resolved relative to it.
pub fn parse_skill(text: &str, dir: &Path) -> Result<Skill> {
    let (yaml, _body) = split_frontmatter(text);
    if yaml.is_empty() {
        bail!("SKILL.md has no YAML frontmatter (must start with `---`)");
    }
    let raw: RawManifest =
        serde_yaml::from_str(&yaml).with_context(|| "frontmatter is not valid YAML")?;

    let name = raw.name.trim().to_string();
    if !skill_name_re().is_match(&name) {
        bail!(
            "skill name `{name}` is invalid: must match [a-z][a-z0-9_]{{0,62}}[a-z0-9] (lowercase, \
             start with a letter, end with a letter/digit)"
        );
    }
    if name.starts_with('_') {
        bail!("skill names starting with `_` are reserved for templates/docs and are not loaded");
    }
    if RESERVED_SKILL_NAMES.contains(&name.as_str()) {
        bail!("`{name}` is a reserved built-in tool name; pick a different one");
    }

    let risk = match raw.risk.to_ascii_lowercase().as_str() {
        "low" => RiskLevel::Low,
        "medium" | "med" => RiskLevel::Medium,
        "high" => RiskLevel::High,
        other => bail!("risk must be low/medium/high, got `{other}`"),
    };

    let entry_name = raw
        .entry
        .clone()
        .unwrap_or_else(|| default_entry_for(&raw.runtime, &name));
    let entry = dir.join(&entry_name);
    if !entry.exists() {
        bail!(
            "entry script `{}` does not exist under `{}`",
            entry_name,
            dir.display()
        );
    }

    let parameters = raw
        .parameters
        .unwrap_or_else(|| json!({ "type": "object", "properties": {} }));

    Ok(Skill {
        name,
        description: raw.description.trim().to_string(),
        parameters,
        risk,
        runtime: raw.runtime.to_ascii_lowercase(),
        entry,
        dir: dir.to_path_buf(),
        timeout_secs: raw.timeout_secs.clamp(1, 600),
    })
}

fn default_entry_for(runtime: &str, name: &str) -> String {
    match runtime {
        "python" | "python3" => "main.py".into(),
        "bash" | "sh" => "main.sh".into(),
        "node" | "nodejs" | "js" => "main.js".into(),
        _ => format!("{name}.py"),
    }
}

/// Resolve the `python`/`bash`/`node` interpreter for the given runtime spec.
fn resolve_runtime(runtime: &str) -> Result<Command> {
    match runtime {
        "python" | "python3" | "py" => {
            // Prefer the on-PATH python3, then python. This is the same Windows-friendly
            // dance the run_command tool uses, but here we keep it simple: first hit wins.
            for cand in ["python3", "python"] {
                if which(cand).is_some() {
                    let mut c = Command::new(cand);
                    c.arg("-u"); // unbuffered, so we get live output for the model
                    return Ok(c);
                }
            }
            bail!("no `python` or `python3` interpreter on PATH");
        }
        "bash" | "sh" => {
            // Re-use the Git Bash / MSYS2 discovery from run_command, but fall back to a plain
            // `bash` lookup so non-Windows users don't see a confusing error.
            #[cfg(windows)]
            {
                for cand in [
                    r"C:\Program Files\Git\bin\bash.exe",
                    r"D:\msys64\usr\bin\bash.exe",
                    r"C:\msys64\usr\bin\bash.exe",
                ] {
                    if std::path::Path::new(cand).exists() {
                        let mut c = Command::new(cand);
                        c.arg("-lc");
                        return Ok(c);
                    }
                }
            }
            let mut c = Command::new("bash");
            c.arg("-lc");
            Ok(c)
        }
        "node" | "nodejs" | "js" => {
            for cand in ["node", "nodejs"] {
                if which(cand).is_some() {
                    return Ok(Command::new(cand));
                }
            }
            bail!("no `node` interpreter on PATH");
        }
        other => bail!("unsupported runtime `{other}` (use python, bash, or node)"),
    }
}

fn which(cmd: &str) -> Option<PathBuf> {
    let exts: &[&str] = if cfg!(windows) {
        &["", ".exe", ".bat", ".cmd"]
    } else {
        &[""]
    };
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        for ext in exts {
            let candidate = dir.join(format!("{cmd}{ext}"));
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

/// Discover skills across the supplied search roots. The first occurrence of a given name wins
/// per-root; across roots, **later roots override earlier ones**, so callers pass user-root first
/// and project-root second.
pub fn discover_skills(roots: &[PathBuf]) -> Vec<Skill> {
    let mut out: HashMap<String, Skill> = HashMap::new();
    let mut errors: Vec<String> = Vec::new();
    for root in roots {
        let Ok(rd) = std::fs::read_dir(root) else {
            continue;
        };
        for entry in rd.flatten() {
            let p = entry.path();
            if !p.is_dir() {
                continue;
            }
            let manifest = p.join("SKILL.md");
            if !manifest.is_file() {
                continue;
            }
            let text = match std::fs::read_to_string(&manifest) {
                Ok(t) => t,
                Err(e) => {
                    errors.push(format!("read {}: {e}", manifest.display()));
                    continue;
                }
            };
            match parse_skill(&text, &p) {
                Ok(s) => {
                    out.insert(s.name.clone(), s);
                }
                Err(e) => {
                    errors.push(format!("parse {}: {e:#}", manifest.display()));
                }
            }
        }
    }
    for e in errors {
        crate::logger::warn("skills", &e);
    }
    let mut v: Vec<Skill> = out.into_values().collect();
    v.sort_by(|a, b| a.name.cmp(&b.name));
    v
}

/// A `Tool` adapter that runs a [`Skill`] as a child process.
pub struct SkillTool {
    inner: Arc<Skill>,
}

impl SkillTool {
    pub fn new(skill: Skill) -> Self {
        Self {
            inner: Arc::new(skill),
        }
    }
}

#[async_trait]
impl Tool for SkillTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            r#type: "function".into(),
            function: FunctionSpec {
                name: self.inner.name.clone(),
                description: self.inner.description.clone(),
                parameters: self.inner.parameters.clone(),
            },
        }
    }

    fn metadata(&self) -> ToolMetadata {
        ToolMetadata {
            category: "skill".into(),
            risk_level: self.inner.risk.clone(),
            ..Default::default()
        }
    }

    async fn call(&self, args: Value) -> Result<Value> {
        let mut cmd = resolve_runtime(&self.inner.runtime)?;
        // The entry is always run as a path, never piped into bash, so the runtime sees a
        // real script file. The script gets the call args as JSON on stdin.
        cmd.arg(&self.inner.entry);
        cmd.current_dir(&self.inner.dir);
        cmd.stdin(Stdio::piped());
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());
        // Inherit a minimal, predictable environment. PATH is needed for the script itself to
        // find helpers; the rest is up to the OS.
        cmd.env_remove("OPENAI_API_KEY");
        cmd.env_remove("DEEPSEEK_API_KEY");
        cmd.env_remove("ANTHROPIC_API_KEY");

        let mut child = cmd.spawn().with_context(|| {
            format!(
                "failed to spawn `{} {}`",
                cmd.as_std().get_program().to_string_lossy(),
                self.inner.entry.display()
            )
        })?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow!("child stdin was not piped"))?;
        let mut stdin = stdin;
        let bytes = serde_json::to_vec(&args).unwrap_or_else(|_| b"{}".to_vec());
        // Best-effort write; if the script closes stdin early (it shouldn't, but just in case)
        // we ignore the broken-pipe error.
        let _ = stdin.write_all(&bytes).await;
        drop(stdin);

        let timeout = Duration::from_secs(self.inner.timeout_secs as u64);
        let output = match tokio::time::timeout(timeout, child.wait_with_output()).await {
            Ok(Ok(o)) => o,
            Ok(Err(e)) => bail!("skill `{}` crashed: {e}", self.inner.name),
            Err(_) => bail!(
                "skill `{}` exceeded the {}s timeout",
                self.inner.name,
                self.inner.timeout_secs
            ),
        };

        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();

        if !output.status.success() {
            let code = output.status.code().unwrap_or(-1);
            // Surface a structured error so the model can react (and the user sees it).
            bail!(
                "skill `{}` exited with code {code}\n--- stderr ---\n{stderr}",
                self.inner.name
            );
        }

        // The script either produced JSON (preferred — models can quote/inspect it) or plain
        // text, which we wrap so the model still has a single, structured result to react to.
        let trimmed = stdout.trim();
        let body: Value = if trimmed.is_empty() {
            Value::Null
        } else if let Ok(v) = serde_json::from_str::<Value>(trimmed) {
            v
        } else {
            json!({ "output": trimmed })
        };

        let mut wrapper = serde_json::Map::new();
        wrapper.insert("result".into(), body);
        if !stderr.trim().is_empty() {
            wrapper.insert("_stderr".into(), Value::String(stderr));
        }
        Ok(Value::Object(wrapper))
    }
}

/// Where a skill's source files should live.
pub fn user_skills_dir() -> Option<PathBuf> {
    Some(dirs::home_dir()?.join(".openworker").join("skills"))
}

pub fn project_skills_dir() -> PathBuf {
    // The project root is the directory the user launched openworker from. Anchoring skills
    // here makes them travel with the repo (they can be committed, code-reviewed, shared).
    std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join(".openworker")
        .join("skills")
}

/// Default search roots, user first, project second so project wins on collision.
pub fn default_search_roots() -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Some(u) = user_skills_dir() {
        v.push(u);
    }
    v.push(project_skills_dir());
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frontmatter_splits_cleanly() {
        let s = "---\nname: foo\n---\nbody here\n";
        let (y, b) = split_frontmatter(s);
        assert!(y.contains("name: foo"));
        assert!(b.contains("body here"));
    }

    #[test]
    fn no_frontmatter_passes_through() {
        let s = "just a body, no front matter\n";
        let (y, b) = split_frontmatter(s);
        assert!(y.is_empty());
        assert!(b.contains("just a body"));
    }

    #[test]
    fn parses_minimal_skill() {
        let dir = tempdir_no_dep();
        std::fs::write(dir.join("main.py"), "print('hi')").unwrap();
        let text = "---\nname: my_skill\ndescription: does things\nrisk: low\nruntime: python\n---\n";
        let s = parse_skill(text, &dir).unwrap();
        assert_eq!(s.name, "my_skill");
        assert_eq!(s.runtime, "python");
        assert_eq!(s.risk, RiskLevel::Low);
    }

    #[test]
    fn rejects_reserved_name() {
        let dir = tempdir_no_dep();
        std::fs::write(dir.join("main.py"), "").unwrap();
        let text = "---\nname: read_file\ndescription: bad\n---\n";
        assert!(parse_skill(text, &dir).is_err());
    }

    #[test]
    fn rejects_template_name() {
        let dir = tempdir_no_dep();
        std::fs::write(dir.join("main.py"), "").unwrap();
        let text = "---\nname: _template\ndescription: docs\n---\n";
        assert!(parse_skill(text, &dir).is_err());
    }

    #[test]
    fn rejects_missing_entry() {
        let dir = tempdir_no_dep();
        let text = "---\nname: hello\ndescription: x\n---\n";
        let err = parse_skill(text, &dir).unwrap_err().to_string();
        assert!(err.contains("entry script"));
    }

    /// End-to-end discovery check against the `hello` sample skill shipped in
    /// `openworker/.openworker/skills/hello/`. This is more of a smoke test — if the file
    /// gets renamed or moved, that's the failure mode we want to catch.
    #[test]
    fn discovers_shipped_hello_skill() {
        // The skill lives two directories up from the openworker-rs crate root.
        let here = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let hello_dir = here
            .parent()
            .unwrap()
            .join(".openworker")
            .join("skills")
            .join("hello");
        if !hello_dir.exists() {
            // Skip silently if the sample has been removed; we don't want a missing
            // example to break CI on every refactor.
            eprintln!("skipping: sample skill not present at {}", hello_dir.display());
            return;
        }
        let found = discover_skills(&[hello_dir.parent().unwrap().to_path_buf()]);
        let names: Vec<&str> = found.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"hello"), "expected to find `hello`, got {names:?}");
    }

    /// Tiny in-test "tempdir" that doesn't pull in the `tempfile` crate: just use a uniquely
    /// named subdir under the OS temp dir and clean it up at the end.
    fn tempdir_no_dep() -> PathBuf {
        let base = std::env::temp_dir().join(format!(
            "ow-skills-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&base).unwrap();
        base
    }

    #[tokio::test]
    async fn runs_skill_via_stdin_stdout() {
        // The runner is the contract: stdin JSON in, stdout JSON out, exit 0. If this
        // breaks, every file-based skill breaks, so we hit the live `hello` sample rather
        // than a synthetic one to exercise the real entry path.
        let here = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let hello_dir = here
            .parent()
            .unwrap()
            .join(".openworker")
            .join("skills")
            .join("hello");
        if !hello_dir.exists() {
            eprintln!("skipping: hello sample not present at {}", hello_dir.display());
            return;
        }
        let text = std::fs::read_to_string(hello_dir.join("SKILL.md")).unwrap();
        let s = parse_skill(&text, &hello_dir).unwrap();
        let tool = SkillTool::new(s);
        let out = tool
            .call(serde_json::json!({ "message": "from test", "times": 2 }))
            .await
            .unwrap();
        let result = out.get("result").expect("wrapper has `result`");
        let echo = result.get("echo").and_then(|v| v.as_array()).expect("echo array");
        assert_eq!(echo.len(), 2);
        assert_eq!(echo[0], "from test");
    }
}
