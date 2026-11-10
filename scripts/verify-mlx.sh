#!/bin/bash
# Optional live MLX end-to-end: setup-mlx plus a real remember/recall cycle on
# the Metal (or CPU) sidecar path. Downloads ~80 MB of model weights into the
# isolated root; nothing outside it is touched.
set -euo pipefail

RB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${RUSTY_BRAIN_MLX_E2E:-}" != "1" ]]; then
    printf 'SKIP: MLX e2e is gated; set RUSTY_BRAIN_MLX_E2E=1 to run the live\n'
    printf 'download + Metal/CPU embed verification (requires uv and network)\n'
    exit 0
fi

if [[ -z "${VERIFY_ROOT:-}" ]]; then
    VERIFY_ROOT="$(mktemp -d /tmp/rusty-brain-mlx-verify.XXXXXX)"
fi
case "$VERIFY_ROOT" in
    /tmp/rusty-brain-mlx-verify.*) ;;
    *) printf 'VERIFY_ROOT must be an isolated /tmp/rusty-brain-mlx-verify.* directory\n' >&2; exit 2 ;;
esac
[[ "$VERIFY_ROOT" != "$HOME" && "$VERIFY_ROOT" != "/" ]]
mkdir -p "$VERIFY_ROOT/work/project" "$VERIFY_ROOT/install"
export VERIFY_ROOT
export RUSTY_BRAIN_VERIFY_ROOT="$VERIFY_ROOT"
export RUSTY_BRAIN_HOME="$VERIFY_ROOT/home"
export RUSTY_BRAIN_PROJECT="rb-mlx-e2e"
# Measured on this machine: the HF Xet transport stalls; plain HTTP does not.
export HF_HUB_DISABLE_XET=1
BIN="$VERIFY_ROOT/install/bin/rusty-brain"

printf 'VERIFY_ROOT=%s\n' "$VERIFY_ROOT"
printf 'Installing the current native binary into the isolated root\n'
cargo install --offline --debug --force --path "$RB" --root "$VERIFY_ROOT/install"
[[ -x "$BIN" ]]

printf 'Running rusty-brain setup-mlx (sidecar + pinned venv + ~80 MB model)\n'
cd "$VERIFY_ROOT/work/project"
"$BIN" setup-mlx
[[ -f "$RUSTY_BRAIN_HOME/sidecar/mlx_serve.py" ]]
[[ -x "$RUSTY_BRAIN_HOME/mlx-venv/bin/python" ]]
[[ -s "$RUSTY_BRAIN_HOME/models/mlx-modernbert-4bit/config.json" ]]
WEIGHTS_BYTES="$(stat -f%z "$RUSTY_BRAIN_HOME/models/mlx-modernbert-4bit/model.safetensors")"
# The published artifact is ~80 MiB; setup already matched it against the
# server's content-length, so anything far below that means truncation.
[[ "$WEIGHTS_BYTES" -gt 67108864 ]] || { printf 'weights too small: %s bytes\n' "$WEIGHTS_BYTES" >&2; exit 1; }
printf 'model weights verified: %s bytes\n' "$WEIGHTS_BYTES"

ID_A="$("$BIN" --agent mlx-agent --session mlx-session remember 'DuckDB instances are reused per customer' --tags decision)"
ID_B="$("$BIN" --agent mlx-agent --session mlx-session remember 'Parquet exports are derived, write-once snapshots' --tags principle)"
ID_C="$("$BIN" --agent other-agent --session other-session remember 'Ollama serves the local agent model' --tags ollama)"
printf 'remembered: %s %s %s\n' "$ID_A" "$ID_B" "$ID_C"

# `--agent self` resolves the calling agent from the env (a global --agent
# would collide with the flag), mirroring how harness mounts work.
RUSTY_BRAIN_AGENT=mlx-agent "$BIN" recall 'how do we reuse warm database instances' --agent self --limit 5 \
    > "$VERIFY_ROOT/recall-self.json"
"$BIN" recall 'how do we reuse warm database instances' --limit 5 \
    > "$VERIFY_ROOT/recall-all.json"
uv run --no-project python -I - "$VERIFY_ROOT" "$ID_A" "$ID_C" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
def ids(name):
    return [hit["memory"]["id"] for hit in json.load(open(root / name, encoding="utf-8"))]
self_ids = ids("recall-self.json")
all_ids = ids("recall-all.json")
assert sys.argv[2] in self_ids[:3], f"self recall A rank failed: {self_ids}"
assert sys.argv[2] in self_ids, "self recall missed A"
assert sys.argv[3] not in self_ids, "self recall must exclude the other agent"
assert sys.argv[3] in all_ids and sys.argv[2] in all_ids, f"all recall missed rows: {all_ids}"
print("recall: self-scoped hit in top-3 and other-agent excluded; all includes both")
PY

"$BIN" stats > "$VERIFY_ROOT/stats-before.json"
uv run --no-project python -I "$VERIFY_ROOT/stats-before.json" <<'PY'
import json, sys
stats = json.load(open(sys.argv[1], encoding="utf-8"))
assert stats["model"] == "mlx-modernbert-4bit", stats
assert stats["memories"] == 3, stats
print("stats: 3 memories on the MLX default model")
PY

printf 'Re-embedding to BGESmallENV15Q (768 -> 384) and switching back\n'
"$BIN" reembed BGESmallENV15Q > "$VERIFY_ROOT/reembed.stdout"
"$BIN" use-model mlx-modernbert-4bit > "$VERIFY_ROOT/use-model.stdout"
"$BIN" stats > "$VERIFY_ROOT/stats-after.json"
uv run --no-project python -I "$VERIFY_ROOT/stats-after.json" "$VERIFY_ROOT/reembed.stdout" <<'PY'
import json, sys
stats = json.load(open(sys.argv[1], encoding="utf-8"))
reembed = open(sys.argv[2], encoding="utf-8").read()
assert "3 memories" in reembed and "BGESmallENV15Q" in reembed, reembed
assert stats["model"] == "mlx-modernbert-4bit" and stats["memories"] == 3, stats
assert set(stats["models"]) == {"mlx-modernbert-4bit", "BGESmallENV15Q"}, stats
print("reembed + use-model roundtrip: both vector sets retained; pointer back on MLX")
PY

printf 'Exporting the MLX-wide snapshot and reading it back through DuckDB\n'
# The slug is cwd-derived; ask the binary for the authoritative brain path.
PROJECT_DB="$("$BIN" path | sed -n 's/^brain: *//p')"
[[ -f "$PROJECT_DB" ]]
"$BIN" export > "$VERIFY_ROOT/export.stdout"
PARSED="$(uv run --no-project python -I -c 'import re,sys; m=re.fullmatch(r"exported (\d+) memories to (.+)", open(sys.argv[1]).read().strip()); assert m; print(m[2] + "|" + m[1])' "$VERIFY_ROOT/export.stdout")"
IFS='|' read -r EXPORT_PATH EXPORT_COUNT <<< "$PARSED"
[[ "$EXPORT_COUNT" == "3" && -f "$EXPORT_PATH" ]]
uv run --isolated "$RB/scripts/verify_parquet.py" --db "$PROJECT_DB" --parquet "$EXPORT_PATH" --reported-count 3

printf 'MLX VERIFIED: setup, live Metal/CPU embedding, recall scoping, reembed/use-model roundtrip, and 768-wide Parquet all passed\n'
printf 'evidence root: %s\n' "$VERIFY_ROOT"
