#!/usr/bin/env python3
"""Minimal stdio MCP server exposing a single `echo` tool, for the openworker-rs smoke test.

Speaks JSON-RPC 2.0, line-delimited, over stdin/stdout — the exact transport the Rust MCP
client uses. It handles `initialize`, `tools/list`, and `tools/call`, and ignores
notifications (messages without an `id`). Exits on EOF (when the parent process closes its
end of the pipe), so spawned servers never leak.

The `echo` tool returns `echo: <text>`, which is what the smoke test asserts on.

Usage:  python tests/fixtures/mock_mcp.py
"""

import json
import sys

ECHO_TOOL = {
    "name": "echo",
    "description": "Echo back the provided text.",
    "inputSchema": {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
}


def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def handle(request: dict):
    method = request.get("method")
    req_id = request.get("id")

    if method == "initialize":
        send(
            {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "serverInfo": {"name": "mock", "version": "1.0.0"},
                },
            }
        )
    elif method == "tools/list":
        send({"jsonrpc": "2.0", "id": req_id, "result": {"tools": [ECHO_TOOL]}})
    elif method == "tools/call":
        params = request.get("params", {})
        name = params.get("name")
        arguments = params.get("arguments", {})
        if name == "echo":
            text = arguments.get("text", "")
            send(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": "echo: " + str(text)}]
                    },
                }
            )
        else:
            send(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": "unknown tool: " + str(name)},
                }
            )
    # Notifications and unknown methods are ignored silently.


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except Exception:
            continue
        # Only respond to actual requests (those carrying an id); skip notifications.
        if "id" not in request:
            continue
        handle(request)


if __name__ == "__main__":
    main()
