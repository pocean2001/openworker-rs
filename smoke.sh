#!/usr/bin/env bash
# End-to-end smoke test — no API key, no network, no real model.
#
#   ./smoke.sh              # uses target/debug (builds if needed)
#   ./smoke.sh --release    # uses target/release
#
# Spins up two local fixtures:
#   tests/fixtures/mock_llm.py  — an OpenAI-compatible /v1/chat/completions server that
#                                 asks for a list_dir tool call, then answers in text
#   tests/fixtures/mock_mcp.py  — a stdio MCP server exposing one `echo` tool
#
# and asserts the whole loop works: model -> tool call -> tool execution -> tool result
# -> final answer, plus MCP discovery/invocation, automations, and config validation.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && (pwd -W 2>/dev/null || pwd))"
cd "$ROOT"

PROFILE="debug"
CARGO_ARGS=()
if [ "${1:-}" = "--release" ]; then
  PROFILE="release"
  CARGO_ARGS=(--release)
fi

PY="${PYTHON:-python}"
command -v "$PY" >/dev/null 2>&1 || PY="python3"
command -v "$PY" >/dev/null 2>&1 || { echo "error: python not found (set PYTHON=...)" >&2; exit 1; }

PORT="${PORT:-8111}"
BIN="$ROOT/target/$PROFILE/openworker.exe"
[ -x "$BIN" ] || BIN="$ROOT/target/$PROFILE/openworker"

./build.sh "${CARGO_ARGS[@]}" >/dev/null || { echo "build failed" >&2; exit 1; }

# Keep the scratch dir inside the project: the binary is a native Windows exe and cannot
# resolve MSYS-style paths like /tmp/tmp.XXXX that `mktemp -d` hands out under Git Bash.
WORK="$ROOT/target/smoke-$$"
mkdir -p "$WORK"
cleanup() {
  [ -n "${LLM_PID:-}" ] && kill "$LLM_PID" 2>/dev/null
  rm -rf "$WORK"
}
trap cleanup EXIT

"$PY" tests/fixtures/mock_llm.py "$PORT" &
LLM_PID=$!
sleep 2

cat > "$WORK/smoke.toml" <<TOML
mode = "auto"
instructions = "You are a smoke-test agent."

[model]
provider = "custom"
model = "mock-model"
base_url = "http://127.0.0.1:$PORT/v1"
api_key = "test-key"

[[mcp_servers]]
name = "mock"
command = "$PY"
args = ["$ROOT/tests/fixtures/mock_mcp.py"]

[[automations]]
name = "hourly-ping"
prompt = "Say hello."
cron = "0 * * * *"
TOML

printf '[model]\nprovider = "openai"\nmdoel = "typo"\n' > "$WORK/bad.toml"

FAILED=0
check() { # check <label> <expected-substring> <actual>
  if printf '%s' "$3" | grep -qF -- "$2"; then
    echo "  PASS  $1"
  else
    echo "  FAIL  $1"
    echo "        expected to contain: $2"
    echo "        got: $(printf '%s' "$3" | head -c 400)"
    FAILED=1
  fi
}

echo "openworker smoke test ($PROFILE)"

out=$("$BIN" --config "$WORK/smoke.toml" run --mode auto --session "smoke-$$" --prompt "list the current directory" 2>&1)
check "agent loop calls a tool"      "list_dir" "$out"
check "agent loop finishes the turn" "completed" "$out"

out=$("$BIN" --config "$WORK/smoke.toml" mcp list 2>&1)
check "MCP tool discovery" "mcp__mock__echo" "$out"

out=$("$BIN" --config "$WORK/smoke.toml" mcp call mock echo '{"text":"ping"}' 2>&1)
check "MCP tool invocation" "echo: ping" "$out"

out=$("$BIN" --config "$WORK/smoke.toml" automation list 2>&1)
check "automation listing" "hourly-ping" "$out"

out=$("$BIN" --config "$WORK/bad.toml" run --prompt "hi" 2>&1)
check "config typos are rejected" "unknown field" "$out"

if [ "$FAILED" -eq 0 ]; then
  echo "all checks passed"
else
  echo "some checks FAILED"
fi
exit "$FAILED"
