#!/usr/bin/env bash
# Build openworker-rs with the local isolated Rust toolchain + MSYS2 MinGW-w64.
#
#   ./build.sh                 -> cargo build
#   ./build.sh --release       -> cargo build --release
#   ./build.sh check           -> cargo check
#   ./build.sh run -- run "hi" -> cargo run -- run "hi"
#
# WHY THIS SCRIPT EXISTS
# ----------------------
# .cargo/config.toml pins the linker/ar/dlltool by absolute path, but that is not
# sufficient on its own: MSYS2's gcc.exe spawns cc1.exe out of
#   D:\msys64\mingw64\lib\gcc\x86_64-w64-mingw32\<ver>\
# and cc1.exe loads libisl / libmpc / libmpfr / libgmp / zlib from
#   D:\msys64\mingw64\bin
# which is a *different* directory, so Windows' DLL search only finds them when
# that bin directory is on PATH. Cargo's `[env] PATH` does NOT help here: cargo
# rebuilds PATH from the parent process before running build scripts, discarding
# the configured value. Hence PATH must be exported at the shell level -- here.

set -euo pipefail

# Windows-style paths: rustup.exe / cargo.exe are native Windows binaries and cannot
# read MSYS-style "/c/..." paths, which is what plain `pwd` yields under Git Bash.
# `pwd -W` emits "C:/..."; on a real POSIX shell it does not exist, so fall back.
winpwd() { pwd -W 2>/dev/null || pwd; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && winpwd)"
TOOLCHAIN="$(cd "$ROOT/.." && winpwd)/.toolchain"

# --- Rust toolchain (isolated; falls back to whatever is on PATH) -------------
if [ -x "$TOOLCHAIN/cargo/bin/cargo" ]; then
  export RUSTUP_HOME="$TOOLCHAIN/rustup"
  export CARGO_HOME="$TOOLCHAIN/cargo"
  CARGO="$TOOLCHAIN/cargo/bin/cargo"
else
  CARGO="$(command -v cargo || true)"
fi
if [ -z "${CARGO:-}" ]; then
  echo "error: no cargo found (looked in $TOOLCHAIN/cargo/bin and PATH)" >&2
  exit 1
fi

# --- MinGW-w64 (gcc for `ring`, dlltool for `windows-sys`) --------------------
MINGW="${MINGW_BIN:-}"
if [ -z "$MINGW" ]; then
  for cand in /d/msys64/mingw64/bin /c/msys64/mingw64/bin /c/mingw64/bin; do
    [ -x "$cand/gcc.exe" ] && MINGW="$cand" && break
  done
fi
if [ -n "$MINGW" ]; then
  export PATH="$MINGW:$PATH"
elif ! command -v gcc >/dev/null 2>&1; then
  echo "error: gcc not found. Install MSYS2 MinGW-w64 or set MINGW_BIN=/path/to/mingw64/bin" >&2
  exit 1
fi

# --- Dispatch: first arg may be a cargo subcommand ----------------------------
SUB="build"
if [ $# -gt 0 ]; then
  case "$1" in
    build|check|run|test|clippy|fmt|clean|doc)
      SUB="$1"
      shift
      ;;
  esac
fi

cd "$ROOT"
exec "$CARGO" "$SUB" "$@"
