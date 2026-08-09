//! Approval policy — ports `coworker/permissions.py` (trimmed to the core).
//!
//! The engine asks the [`PermissionEngine`] whether a tool call may run. In `Auto` mode
//! everything is allowed; in `Interactive` mode low-risk reads run freely while writes /
//! shell / high-risk calls surface a [`PermissionRequest`] to the [`Approver`]; `Plan` and
//! `Discuss` are read-only and disable tool execution entirely.

use std::collections::HashSet;

use serde_json::Value;

/// Static risk classification of a tool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
}

/// Agent operating mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Ask before consequential actions.
    Interactive,
    /// Approve everything (headless / unattended).
    Auto,
    /// Read-only; never execute a tool.
    Plan,
    /// Read-only discussion.
    Discuss,
}

impl Mode {
    pub fn from_str(s: &str) -> Mode {
        match s.to_ascii_lowercase().as_str() {
            "auto" => Mode::Auto,
            "plan" => Mode::Plan,
            "discuss" => Mode::Discuss,
            _ => Mode::Interactive,
        }
    }
}

/// Per-tool metadata used by the permission engine and surfaced in tool cards.
#[derive(Debug, Clone)]
pub struct ToolMetadata {
    pub category: String,
    pub risk_level: RiskLevel,
    pub requires_approval: bool,
    pub capabilities: Vec<String>,
}

impl Default for ToolMetadata {
    fn default() -> Self {
        ToolMetadata {
            category: "builtin".into(),
            risk_level: RiskLevel::Low,
            requires_approval: false,
            capabilities: vec![],
        }
    }
}

/// A request for the user's blessing before a tool runs.
#[derive(Debug, Clone)]
pub struct PermissionRequest {
    pub tool_name: String,
    pub arguments: Value,
    pub metadata: Option<ToolMetadata>,
    pub reason: String,
    pub tool_call_id: Option<String>,
}

/// The outcome of an approval interaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApprovalOutcome {
    /// Allow this once.
    Once,
    /// Allow every call to this tool for the rest of the session.
    AlwaysTool,
    /// Allow every call to this exact command for the rest of the session.
    AlwaysCommand,
    /// Deny.
    Deny,
}

/// Decides approvals out-of-band (console prompt, always-allow, always-deny, ...).
pub trait Approver: Send + Sync {
    fn approve(&self, req: &PermissionRequest) -> ApprovalOutcome;
}

/// Interactive console approver.
pub struct ConsoleApprover;
impl Approver for ConsoleApprover {
    fn approve(&self, req: &PermissionRequest) -> ApprovalOutcome {
        use std::io::Write;
        // The arguments are already shown by the `ToolProposed` event; repeating them here
        // would print the same JSON twice for every approval.
        print!(
            "    approve {}? [y = once / a = always this tool / d = deny]: ",
            req.tool_name
        );
        let _ = std::io::stdout().flush();
        let mut s = String::new();
        let _ = std::io::stdin().read_line(&mut s);
        match s.trim().to_ascii_lowercase().as_str() {
            "y" | "yes" => ApprovalOutcome::Once,
            "a" => ApprovalOutcome::AlwaysTool,
            _ => ApprovalOutcome::Deny,
        }
    }
}

/// Headless "approve everything" approver.
pub struct AutoApprover;
impl Approver for AutoApprover {
    fn approve(&self, _req: &PermissionRequest) -> ApprovalOutcome {
        ApprovalOutcome::Once
    }
}

/// "Deny everything" approver.
pub struct DenyApprover;
impl Approver for DenyApprover {
    fn approve(&self, _req: &PermissionRequest) -> ApprovalOutcome {
        ApprovalOutcome::Deny
    }
}

/// The result of [`PermissionEngine::evaluate`].
#[derive(Debug, Clone)]
pub struct Decision {
    pub allowed: bool,
    pub needs_user: bool,
    pub reason: String,
}

/// Holds the active mode plus session-scoped allow lists granted via "always" approvals.
pub struct PermissionEngine {
    pub mode: Mode,
    allow_tools: HashSet<String>,
    allow_commands: HashSet<String>,
}

impl PermissionEngine {
    pub fn new(mode: Mode) -> Self {
        PermissionEngine {
            mode,
            allow_tools: HashSet::new(),
            allow_commands: HashSet::new(),
        }
    }

    pub fn allow_tool_for_session(&mut self, name: &str) {
        self.allow_tools.insert(name.to_string());
    }

    pub fn allow_command_for_session(&mut self, command: &str) {
        self.allow_commands.insert(command.to_string());
    }

    pub fn evaluate(
        &self,
        name: &str,
        args: &Value,
        meta: Option<&ToolMetadata>,
    ) -> Decision {
        match self.mode {
            Mode::Auto => Decision {
                allowed: true,
                needs_user: false,
                reason: "auto mode".into(),
            },
            Mode::Plan | Mode::Discuss => Decision {
                allowed: false,
                needs_user: false,
                reason: "read-only mode; tool execution is disabled".into(),
            },
            Mode::Interactive => {
                if self.allow_tools.contains(name) {
                    return Decision {
                        allowed: true,
                        needs_user: false,
                        reason: "allowed for session".into(),
                    };
                }
                let meta = meta.cloned().unwrap_or_default();
                if name == "run_command" {
                    if let Some(cmd) = args.get("command").and_then(|v| v.as_str()) {
                        if self.allow_commands.contains(cmd) {
                            return Decision {
                                allowed: true,
                                needs_user: false,
                                reason: "allowed command for session".into(),
                            };
                        }
                    }
                }
                let needs = meta.requires_approval || meta.risk_level == RiskLevel::High;
                if needs {
                    Decision {
                        allowed: false,
                        needs_user: true,
                        reason: format!(
                            "{} requires approval (risk: {:?})",
                            name, meta.risk_level
                        ),
                    }
                } else {
                    Decision {
                        allowed: true,
                        needs_user: false,
                        reason: "low-risk".into(),
                    }
                }
            }
        }
    }
}
