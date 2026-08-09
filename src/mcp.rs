//! MCP client — ports `coworker/mcp/client.py` + `coworker/mcp/tools.py`.
//!
//! A thin, dependency-free JSON-RPC 2.0 client over a server's stdio transport. It performs
//! `initialize` → `tools/list` → `tools/call` and bridges each remote tool into the
//! [`ToolRegistry`] under the name `mcp__<server>__<tool>` (sanitized for the
//! `[A-Za-z0-9_-]{1,64}` rule OpenAI enforces on function names).

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use regex::Regex;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;

use crate::permissions::ToolMetadata;
use crate::provider::{FunctionSpec, ToolSpec};
use crate::tools::{Tool, ToolRegistry};

/// A configured MCP server (mirrors `coworker/mcp/config.py::MCPServerDef`).
#[derive(Debug, Clone, Deserialize)]
pub struct McpServerDef {
    pub name: String,
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: HashMap<String, String>,
    #[serde(default)]
    pub include_tools: Option<Vec<String>>,
    #[serde(default)]
    pub exclude_tools: Option<Vec<String>>,
    #[serde(default)]
    pub requires_approval: bool,
}

/// A tool advertised by an MCP server.
pub struct McpToolInfo {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

/// Sanitized `mcp__<server>__<tool>` registry name.
///
/// The doubled separator is deliberate: server and tool names may themselves contain
/// single underscores, so `mcp__github_create_issue` would be ambiguous, while
/// `mcp__github__create_issue` splits cleanly.
///
/// Truncation is done on a char boundary — sanitization maps every non-ASCII byte to
/// `_`, but slicing blindly at 64 would still be a latent panic if that ever changes.
pub fn tool_name(server: &str, tool: &str) -> String {
    let re = Regex::new(r"[^a-zA-Z0-9_-]").unwrap();
    let base = format!(
        "mcp__{}__{}",
        re.replace_all(server, "_"),
        re.replace_all(tool, "_")
    );
    if base.len() > 64 {
        let mut end = 64;
        while end > 0 && !base.is_char_boundary(end) {
            end -= 1;
        }
        base[..end].to_string()
    } else {
        base
    }
}

#[cfg(test)]
mod tests {
    use super::tool_name;

    #[test]
    fn separates_server_and_tool_unambiguously() {
        assert_eq!(tool_name("github", "create_issue"), "mcp__github__create_issue");
    }

    #[test]
    fn sanitizes_illegal_characters() {
        assert_eq!(tool_name("my server", "do/thing"), "mcp__my_server__do_thing");
    }

    #[test]
    fn truncates_long_names_without_panicking() {
        let n = tool_name(&"s".repeat(50), &"t".repeat(50));
        assert_eq!(n.len(), 64);
    }
}

struct Inner {
    /// Owning handle to the MCP server process. Never read, but dropping it would
    /// orphan (and on some platforms kill) the child, so it must be kept alive for
    /// as long as we hold its stdin/stdout pipes.
    #[allow(dead_code)]
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_id: u64,
}

impl Inner {
    async fn write_line(&mut self, line: &str) -> Result<()> {
        self.stdin.write_all(line.as_bytes()).await?;
        self.stdin.write_all(b"\n").await?;
        self.stdin.flush().await?;
        Ok(())
    }

    async fn request(&mut self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id;
        self.next_id += 1;
        let msg = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params
        });
        self.write_line(&serde_json::to_string(&msg)?).await?;
        loop {
            let mut buf = String::new();
            let n = self.stdout.read_line(&mut buf).await?;
            if n == 0 {
                return Err(anyhow!("MCP server closed the connection"));
            }
            let line = buf.trim();
            if line.is_empty() {
                continue;
            }
            let v: Value = match serde_json::from_str(line) {
                Ok(v) => v,
                Err(_) => continue,
            };
            // Skip server-initiated notifications (no id).
            if v.get("id").is_none() {
                continue;
            }
            if v.get("id") == Some(&json!(id)) {
                if let Some(err) = v.get("error") {
                    return Err(anyhow!("MCP error: {}", err));
                }
                return Ok(v.get("result").cloned().unwrap_or(Value::Null));
            }
        }
    }

    async fn notify(&mut self, method: &str, params: Value) -> Result<()> {
        let msg = json!({ "jsonrpc": "2.0", "method": method, "params": params });
        self.write_line(&serde_json::to_string(&msg)?).await
    }
}

/// A live MCP connection. Cheap to clone (shares the underlying process via `Arc`).
#[derive(Clone)]
pub struct McpClient {
    inner: Arc<Mutex<Inner>>,
}

impl McpClient {
    /// Spawn the server process and complete the MCP handshake.
    pub async fn connect(def: &McpServerDef) -> Result<McpClient> {
        let mut cmd = Command::new(&def.command);
        cmd.args(&def.args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::inherit());
        for (k, v) in &def.env {
            cmd.env(k, v);
        }
        let mut child = cmd.spawn().map_err(|e| {
            anyhow!(
                "failed to launch MCP server '{}' ({}): {}",
                def.name,
                def.command,
                e
            )
        })?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow!("MCP server '{}' has no stdin", def.name))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("MCP server '{}' has no stdout", def.name))?;
        let mut inner = Inner {
            child,
            stdin,
            stdout: BufReader::new(stdout),
            next_id: 1,
        };
        inner
            .request(
                "initialize",
                json!({
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": { "name": "openworker-rs", "version": "0.1.0" }
                }),
            )
            .await?;
        inner.notify("notifications/initialized", json!({})).await?;
        Ok(McpClient {
            inner: Arc::new(Mutex::new(inner)),
        })
    }

    /// List the server's tools.
    pub async fn list_tools(&self) -> Result<Vec<McpToolInfo>> {
        let mut g = self.inner.lock().await;
        let res = g.request("tools/list", json!({})).await?;
        let arr = res.get("tools").and_then(|t| t.as_array()).cloned().unwrap_or_default();
        let mut out = Vec::new();
        for t in arr {
            let name = t.get("name").and_then(|x| x.as_str()).unwrap_or("").to_string();
            let description = t
                .get("description")
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .to_string();
            let input_schema = t
                .get("inputSchema")
                .cloned()
                .unwrap_or_else(|| json!({ "type": "object", "properties": {} }));
            out.push(McpToolInfo {
                name,
                description,
                input_schema,
            });
        }
        Ok(out)
    }

    /// Call a tool on the server and flatten the result into something the model can read.
    pub async fn call_tool(&self, name: &str, args: Value) -> Result<Value> {
        let mut g = self.inner.lock().await;
        let res = g
            .request("tools/call", json!({ "name": name, "arguments": args }))
            .await?;
        let content = res
            .get("content")
            .and_then(|c| c.as_array())
            .cloned()
            .unwrap_or_default();
        let mut texts = Vec::new();
        for block in content {
            if let Some(t) = block.get("text").and_then(|x| x.as_str()) {
                texts.push(t.to_string());
            } else {
                let ty = block.get("type").and_then(|x| x.as_str()).unwrap_or("content");
                texts.push(format!("[{}]", ty));
            }
        }
        let body = texts.join("\n");
        let is_error = res.get("isError").and_then(|x| x.as_bool()).unwrap_or(false);
        if is_error {
            Ok(json!({ "error": body }))
        } else if let Some(sc) = res.get("structuredContent") {
            if body.is_empty() {
                Ok(sc.clone())
            } else {
                Ok(Value::String(body))
            }
        } else {
            Ok(Value::String(body))
        }
    }
}

/// A registry-ready wrapper around one remote MCP tool.
struct McpTool {
    client: McpClient,
    remote_name: String,
    spec: ToolSpec,
    metadata: ToolMetadata,
}

#[async_trait]
impl Tool for McpTool {
    fn spec(&self) -> ToolSpec {
        self.spec.clone()
    }
    fn metadata(&self) -> ToolMetadata {
        self.metadata.clone()
    }
    async fn call(&self, args: Value) -> Result<Value> {
        self.client.call_tool(&self.remote_name, args).await
    }
}

fn filtered(tools: Vec<McpToolInfo>, def: &McpServerDef) -> Vec<McpToolInfo> {
    let mut out = tools;
    if let Some(allow) = &def.include_tools {
        let allow: std::collections::HashSet<_> = allow.iter().cloned().collect();
        out.retain(|t| allow.contains(&t.name));
    }
    if let Some(block) = &def.exclude_tools {
        let block: std::collections::HashSet<_> = block.iter().cloned().collect();
        out.retain(|t| !block.contains(&t.name));
    }
    out
}

/// Connect every configured server and return a registry with all of their tools bridged in.
pub async fn connect_mcp_servers(defs: &[McpServerDef]) -> Result<ToolRegistry> {
    let mut reg = ToolRegistry::new();
    for def in defs {
        let client = McpClient::connect(def).await?;
        let tools = filtered(client.list_tools().await?, def);
        for info in tools {
            let name = tool_name(&def.name, &info.name);
            let spec = FunctionSpec {
                name: name.clone(),
                description: info.description.chars().take(1024).collect(),
                parameters: info.input_schema,
            };
            let metadata = ToolMetadata {
                category: "mcp".into(),
                risk_level: crate::permissions::RiskLevel::Medium,
                requires_approval: def.requires_approval,
                capabilities: vec![def.name.clone()],
            };
            let tool = Arc::new(McpTool {
                client: client.clone(),
                remote_name: info.name,
                spec: ToolSpec {
                    r#type: "function".into(),
                    function: spec,
                },
                metadata,
            });
            reg.register(tool);
        }
        println!(
            "[mcp] connected server '{}' ({} tool(s))",
            def.name,
            reg.len()
        );
    }
    Ok(reg)
}
