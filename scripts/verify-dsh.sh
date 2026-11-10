#!/bin/bash
# Real dsh + Ollama acceptance in isolated roots with marker-only credentials.
set -euo pipefail

RB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${VERIFY_ROOT:-}" ]]; then
    VERIFY_ROOT="$(mktemp -d /tmp/rusty-brain-verify.XXXXXX)"
fi
case "$VERIFY_ROOT" in
    /tmp/rusty-brain-verify.*) ;;
    *) printf 'VERIFY_ROOT must be an isolated /tmp/rusty-brain-verify.* directory\n' >&2; exit 2 ;;
esac
[[ "$VERIFY_ROOT" != "$HOME" && "$VERIFY_ROOT" != "/" ]]
export VERIFY_ROOT
export RUSTY_BRAIN_VERIFY_ROOT="$VERIFY_ROOT"
export RUSTY_BRAIN_HOME="$VERIFY_ROOT/home"
export RUSTY_BRAIN_PROJECT="rb-dsh-e2e"
# The dsh session creates this brain; keep the CPU fastembed arm explicit even
# with the MLX registry default (verify_dsh_session.py asserts it).
export RUSTY_BRAIN_MODEL="BGESmallENV15Q"
PROJECT_CWD="$VERIFY_ROOT/work/project"
BIN="$VERIFY_ROOT/install/bin/rusty-brain"
DSH_PID=""
WATCHDOG_PID=""
cleanup() {
    if [[ -n "$WATCHDOG_PID" ]]; then kill "$WATCHDOG_PID" 2>/dev/null || true; fi
    if [[ -n "$DSH_PID" ]]; then kill "$DSH_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT

# Even the structural dump must not inherit the developer's real credentials.
unset DEEPSEEK_API_KEY HF_TOKEN AWS_SECRET_ACCESS_KEY AWS_ACCESS_KEY_ID AWS_SESSION_TOKEN
mkdir -p "$PROJECT_CWD" "$VERIFY_ROOT/install" "$VERIFY_ROOT/os-home"
printf 'VERIFY_ROOT=%s\n' "$VERIFY_ROOT"
printf 'Installing the current native binary into the isolated root\n'
cargo install --offline --debug --force --path "$RB" --root "$VERIFY_ROOT/install"
[[ -x "$BIN" ]]
# The documented PATH includes asdf shims; keep the shim's own launcher isolated too.
ln -sf /opt/homebrew/bin/asdf "$VERIFY_ROOT/install/bin/asdf"
command -v dsh >/dev/null
command -v ollama >/dev/null
dsh --version
ollama --version
ollama show qwen3.8:27b-mlx > "$VERIFY_ROOT/ollama-model.txt"
grep -q 'tools' "$VERIFY_ROOT/ollama-model.txt"
find "$RUSTY_BRAIN_HOME/models" -type f -size +0c | grep -q . >/dev/null || {
    printf 'model cache is not primed; run scripts/verify-local.sh with this VERIFY_ROOT first\n' >&2
    exit 1
}

# The session root begins empty on every run; this path is inside the guarded temp root.
rm -rf -- "$VERIFY_ROOT/sessions"
rm -f -- "$VERIFY_ROOT"/dsh-{config.yml,stdout.log,stderr.log,status,task.txt,evaluation.json,timeout}
cd "$PROJECT_CWD"
dsh --profile headless \
  --patch "$RB/examples/deepseek-harness/ollama-local.cordis.yml" \
  --patch "$RB/examples/deepseek-harness/rusty-brain.cordis.yml" \
  --dump-config > "$VERIFY_ROOT/dsh-config.yml"
uv run --isolated "$RB/scripts/verify_dsh_config.py" \
    --repo "$RB" --dump "$VERIFY_ROOT/dsh-config.yml" \
    --root "$VERIFY_ROOT" --project-cwd "$PROJECT_CWD"

RB_LIVE_TOKEN="RB-DSH-$(uuidgen)"
RB_TEST_SECRET="rb006-unrecognized-dsh-marker-$(uuidgen)"
[[ ${#RB_TEST_SECRET} -ge 16 ]]
cat > "$VERIFY_ROOT/dsh-task.txt" <<EOF
Use only rusty-brain MCP tools mcp__rusty-brain__remember, mcp__rusty-brain__recall, mcp__rusty-brain__stats, mcp__rusty-brain__data_path, and mcp__rusty-brain__forget. In order: (1) remember exact content 'Verification token: $RB_LIVE_TOKEN' with tags 'verification'; (2) recall query '$RB_LIVE_TOKEN' with limit 5 and retain its UUID; (3) call stats; (4) call data_path; (5) forget that exact UUID; (6) recall '$RB_LIVE_TOKEN' again with limit 5 and verify it is absent. Do not use shell, filesystem, web, or subagents. Use exactly six calls total; do not call stats again after forget, and derive the active count from the earlier stats plus successful forget. Final response must contain the token, UUID, active count after forget, and 'verified absent' only if true.
EOF

printf 'Running one real local dsh scenario (900-second timeout)\n'
env -i \
  HOME="$HOME" \
  PATH="$VERIFY_ROOT/install/bin:/Users/rjac/.asdf/shims:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin" \
  TMPDIR="${TMPDIR:-/tmp}" LANG=en_US.UTF-8 \
  RUSTY_BRAIN_HOME="$VERIFY_ROOT/home" \
  RUSTY_BRAIN_PROJECT="rb-dsh-e2e" \
  RUSTY_BRAIN_MODEL="$RUSTY_BRAIN_MODEL" \
  RUSTY_BRAIN_VERIFY_ROOT="$VERIFY_ROOT" \
  RB_TEST_SECRET="$RB_TEST_SECRET" \
  DEEPSEEK_API_KEY="$RB_TEST_SECRET" \
  HF_TOKEN="$RB_TEST_SECRET" \
  AWS_SECRET_ACCESS_KEY="$RB_TEST_SECRET" \
  dsh --profile headless \
    --patch "$RB/examples/deepseek-harness/ollama-local.cordis.yml" \
    --patch "$RB/examples/deepseek-harness/rusty-brain.cordis.yml" \
    "$(cat "$VERIFY_ROOT/dsh-task.txt")" \
    >"$VERIFY_ROOT/dsh-stdout.log" 2>"$VERIFY_ROOT/dsh-stderr.log" &
DSH_PID=$!
(
    /bin/sleep 900
    : > "$VERIFY_ROOT/dsh-timeout"
    kill -TERM "$DSH_PID" 2>/dev/null || true
) &
WATCHDOG_PID=$!
set +e
wait "$DSH_PID"
DSH_STATUS=$?
set -e
DSH_PID=""
kill "$WATCHDOG_PID" 2>/dev/null || true
wait "$WATCHDOG_PID" 2>/dev/null || true
WATCHDOG_PID=""
printf '%s\n' "$DSH_STATUS" > "$VERIFY_ROOT/dsh-status"
[[ ! -e "$VERIFY_ROOT/dsh-timeout" ]] || printf 'dsh timed out after 900 seconds\n' >&2

set +e
uv run --no-project python -I "$RB/scripts/verify_dsh_session.py" \
    --root "$VERIFY_ROOT" --token "$RB_LIVE_TOKEN" \
    --task-file "$VERIFY_ROOT/dsh-task.txt" --output "$VERIFY_ROOT/dsh-evaluation.json"
SESSION_STATUS=$?
set -e

# Deliberate model-initialization failure with marker-only credentials.
set +e
env -i HOME="$VERIFY_ROOT/os-home" PATH="$VERIFY_ROOT/install/bin:/usr/local/bin:/usr/bin:/bin" \
    TMPDIR="${TMPDIR:-/tmp}" LANG=en_US.UTF-8 RUST_BACKTRACE=1 \
    RUSTY_BRAIN_HOME="$VERIFY_ROOT/failing-home" \
    HF_HOME="$VERIFY_ROOT/failing-hf" HF_ENDPOINT=http://127.0.0.1:9 \
    RB_TEST_SECRET="$RB_TEST_SECRET" HF_TOKEN="$RB_TEST_SECRET" \
    DEEPSEEK_API_KEY="$RB_TEST_SECRET" AWS_SECRET_ACCESS_KEY="$RB_TEST_SECRET" \
    "$BIN" --project failing-client remember 'must not persist' \
    >"$VERIFY_ROOT/failing-client.stdout" 2>"$VERIFY_ROOT/failing-client.stderr"
FAILING_STATUS=$?
set -e
[[ "$FAILING_STATUS" -ne 0 ]]

printf '%s\nCaused by: deliberate control\nStack backtrace:\n' "$RB_TEST_SECRET" > "$VERIFY_ROOT/leaking-control.log"
set +e
RB_TEST_SECRET="$RB_TEST_SECRET" uv run --no-project python -I "$RB/scripts/verify_no_secrets.py" \
    "$VERIFY_ROOT/leaking-control.log" >"$VERIFY_ROOT/leaking-control-scan.log" 2>&1
LEAK_STATUS=$?
set -e
[[ "$LEAK_STATUS" -ne 0 ]]
printf 'marker/chain/stack positive control rejected as required\n'

EVIDENCE=(
    "$VERIFY_ROOT/dsh-config.yml" "$VERIFY_ROOT/dsh-stdout.log" "$VERIFY_ROOT/dsh-stderr.log"
    "$VERIFY_ROOT/dsh-task.txt" "$VERIFY_ROOT/dsh-evaluation.json"
    "$VERIFY_ROOT/failing-client.stdout" "$VERIFY_ROOT/failing-client.stderr"
    "$VERIFY_ROOT/leaking-control-scan.log"
)
while IFS= read -r -d '' SESSION_FILE; do EVIDENCE+=("$SESSION_FILE"); done < <(find "$VERIFY_ROOT/sessions" -type f -name 'session*.jsonl' -print0)
if [[ -f "$RB/docs/verification.md" ]]; then EVIDENCE+=("$RB/docs/verification.md"); fi
RB_TEST_SECRET="$RB_TEST_SECRET" uv run --no-project python -I "$RB/scripts/verify_no_secrets.py" "${EVIDENCE[@]}"

if [[ "$DSH_STATUS" -ne 0 || "$SESSION_STATUS" -ne 0 ]]; then
    printf 'H3 FAIL: dsh exit=%s session/status verification exit=%s; evidence=%s\n' \
        "$DSH_STATUS" "$SESSION_STATUS" "$VERIFY_ROOT/dsh-evaluation.json" >&2
    exit 1
fi
printf 'H3 PASS: real local dsh/Ollama ordered lifecycle; evidence=%s\n' "$VERIFY_ROOT/dsh-evaluation.json"
