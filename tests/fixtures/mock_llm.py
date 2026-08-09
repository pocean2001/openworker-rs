#!/usr/bin/env python3
"""Mock OpenAI-compatible chat-completions server for the openworker-rs smoke test.

Serves POST /v1/chat/completions as Server-Sent Events (the exact wire format the Rust
engine streams from). It drives the full agent loop without any real model or network:

  * first tool-capable request  -> asks the engine to call the built-in `list_dir` tool
  * request that already carries a tool result -> returns a plain-text final answer

Recap/compress/wrap-up calls (which the engine sends with an empty `tools` list) are answered
with plain text so they no-op. This makes the mock robust to however many auxiliary provider
calls the engine happens to make.

Usage:  python tests/fixtures/mock_llm.py <port>
"""

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def sse(payload: dict) -> bytes:
    return ("data: " + json.dumps(payload) + "\n\n").encode("utf-8")


def tool_call_chunks():
    # Round 1: request a `list_dir` tool call.
    return [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_smoke_1",
                                "type": "function",
                                "function": {"name": "list_dir", "arguments": "{}"},
                            }
                        ],
                    },
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {"index": 0, "delta": {}, "finish_reason": "tool_calls"}
            ]
        },
    ]


def text_chunks(content: str):
    return [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": content},
                    "finish_reason": None,
                }
            ]
        },
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence default request logging
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8")) if raw else {}
        except Exception:
            body = {}

        messages = body.get("messages", [])
        tools = body.get("tools") or []
        has_tool_result = any(
            isinstance(m, dict) and m.get("role") == "tool" for m in messages
        )

        if tools and not has_tool_result:
            chunks = tool_call_chunks()
        else:
            chunks = text_chunks(
                "The current directory has been listed. Smoke test turn complete."
            )

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        for chunk in chunks:
            self.wfile.write(sse(chunk))
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8111
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
