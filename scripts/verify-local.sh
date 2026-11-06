#!/bin/bash
# Native, isolated end-to-end verification for the CLI, cache, store, and exports.
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
mkdir -p "$VERIFY_ROOT/work/project" "$VERIFY_ROOT/install" "$VERIFY_ROOT/ids"
export VERIFY_ROOT
export RUSTY_BRAIN_VERIFY_ROOT="$VERIFY_ROOT"
export RUSTY_BRAIN_HOME="$VERIFY_ROOT/home"
export RUSTY_BRAIN_PROJECT="rb-local-e2e"
BIN="$VERIFY_ROOT/install/bin/rusty-brain"
PROFILE="$RB/tests/fixtures/no-network.sb"
RUNTIME_PATH="$VERIFY_ROOT/install/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
PROBE_PID=""
cleanup() {
    if [[ -n "$PROBE_PID" ]]; then
        kill "$PROBE_PID" 2>/dev/null || true
        wait "$PROBE_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

printf 'VERIFY_ROOT=%s\n' "$VERIFY_ROOT"
printf 'Installing the current native binary into the isolated root\n'
cargo install --offline --debug --path "$RB" --root "$VERIFY_ROOT/install"
[[ -x "$BIN" ]]

run_rb() {
    local project="$1"
    shift
    env -i HOME="$HOME" PATH="$RUNTIME_PATH" TMPDIR="${TMPDIR:-/tmp}" LANG=en_US.UTF-8 \
        RUSTY_BRAIN_HOME="$RUSTY_BRAIN_HOME" "$BIN" --project "$project" "$@"
}

run_rb_at_home() {
    local home="$1"
    local project="$2"
    shift 2
    env -i HOME="$HOME" PATH="$RUNTIME_PATH" TMPDIR="${TMPDIR:-/tmp}" LANG=en_US.UTF-8 \
        RUSTY_BRAIN_HOME="$home" "$BIN" --project "$project" "$@"
}

sandbox_rb() {
    local project="$1"
    shift
    /usr/bin/sandbox-exec -f "$PROFILE" /usr/bin/env -i \
        HOME="$HOME" PATH="$RUNTIME_PATH" TMPDIR="${TMPDIR:-/tmp}" LANG=en_US.UTF-8 \
        RUSTY_BRAIN_HOME="$RUSTY_BRAIN_HOME" "$BIN" --project "$project" "$@"
}

printf 'Proving the sandbox control succeeds outside and is denied inside\n'
/usr/bin/curl --max-time 15 --silent --show-error --output /dev/null https://huggingface.co \
    >"$VERIFY_ROOT/network-outside.stdout" 2>"$VERIFY_ROOT/network-outside.stderr"
set +e
/usr/bin/sandbox-exec -f "$PROFILE" /usr/bin/curl --max-time 15 --silent --show-error \
    --output /dev/null https://huggingface.co \
    >"$VERIFY_ROOT/network-inside.stdout" 2>"$VERIFY_ROOT/network-inside.stderr"
INSIDE_NETWORK_STATUS=$?
set -e
[[ "$INSIDE_NETWORK_STATUS" -ne 0 ]]
printf 'network control: outside=0 inside=%s (denied)\n' "$INSIDE_NETWORK_STATUS"

KEYS=(A B C D E F G H I J)
CONTENTS=(
    'DuckDB instances are reused per customer'
    'Parquet exports are derived, write-once snapshots'
    'The local brain stores data under the project directory'
    'PostgreSQL connection pooling is not used by rusty-brain'
    'The MCP server communicates over standard input and output'
    'Forgetting a memory is a soft deletion'
    'The embedding model produces 384 values'
    'Use uv to run the DuckDB verification script'
    'Ollama serves the local agent model'
    'deepseek-harness mounts memory through a patch overlay'
)
TAGS=(
    'decision,duckdb'
    'principle,parquet'
    'storage'
    'negative'
    'mcp'
    'lifecycle'
    'embedding'
    'verification'
    'ollama'
    'dsh'
)
IDS=()
printf 'Priming BGESmallENV15Q once and storing exact fixture A-J\n'
for i in "${!KEYS[@]}"; do
    ID="$(cd "$VERIFY_ROOT/work/project" && run_rb "$RUSTY_BRAIN_PROJECT" remember "${CONTENTS[$i]}" --tags "${TAGS[$i]}")"
    [[ "$ID" =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$ ]]
    IDS[$i]="$ID"
    printf '%s\n' "$ID" > "$VERIFY_ROOT/ids/${KEYS[$i]}"
done
find "$RUSTY_BRAIN_HOME/models" -type f -size +0c | grep -q . >/dev/null

cd "$VERIFY_ROOT/work/project"
run_rb "$RUSTY_BRAIN_PROJECT" recall 'customer-specific database instances stay warm' --limit 5 \
    > "$VERIFY_ROOT/semantic.json"
uv run --no-project python -I - "$VERIFY_ROOT/semantic.json" "${IDS[0]}" <<'PY'
import json, sys
hits = json.load(open(sys.argv[1], encoding="utf-8"))
ranks = [i + 1 for i, hit in enumerate(hits) if hit["memory"]["id"] == sys.argv[2]]
assert ranks and ranks[0] <= 3, f"semantic A rank must be <=3: {ranks}"
print(f"semantic A rank={ranks[0]}")
PY

run_rb "$RUSTY_BRAIN_PROJECT" recall 'write-once' --limit 5 > "$VERIFY_ROOT/keyword.json"
uv run --no-project python -I - "$VERIFY_ROOT/keyword.json" "${IDS[1]}" <<'PY'
import json, sys
hits = json.load(open(sys.argv[1], encoding="utf-8"))
ranks = [i + 1 for i, hit in enumerate(hits) if hit["memory"]["id"] == sys.argv[2] and hit["matched_text"]]
assert ranks and ranks[0] <= 5, f"keyword B rank/match failed: {ranks}"
print(f"keyword B rank={ranks[0]} matched_text=true")
PY

run_rb "$RUSTY_BRAIN_PROJECT" recall '" OR * : ( )' --limit 5 > "$VERIFY_ROOT/operator.json"
uv run --no-project python -I - "$VERIFY_ROOT/operator.json" <<'PY'
import copy, json, sys
hits = json.load(open(sys.argv[1], encoding="utf-8"))
def violations(rows):
    return [row["memory"]["id"] for row in rows if row["matched_text"]]
control = copy.deepcopy(hits)
if control:
    control[0]["matched_text"] = True
else:
    control = [{"memory": {"id": "control"}, "matched_text": True}]
assert violations(control), "mutated matched_text positive control was not detected"
assert not violations(hits), f"operator query injected lexical hits: {violations(hits)}"
print("operator query: no injected lexical match; mutated control rejected")
PY

run_rb "$RUSTY_BRAIN_PROJECT" recent --limit 10 > "$VERIFY_ROOT/recent-before.json"
uv run --no-project python -I - "$VERIFY_ROOT/recent-before.json" "$VERIFY_ROOT/ids" <<'PY'
import json, pathlib, sys
rows = json.load(open(sys.argv[1], encoding="utf-8"))
keys = list("ABCDEFGHIJ")
ids = [(pathlib.Path(sys.argv[2]) / key).read_text().strip() for key in keys]
actual = [row["id"] for row in rows]
assert actual == list(reversed(ids)), f"recent must contain A-J newest-first: {actual}"
print("recent before forget: exactly A-J, newest-first")
PY

FORGOTTEN_ID="${IDS[5]}"
[[ "$(run_rb "$RUSTY_BRAIN_PROJECT" forget "$FORGOTTEN_ID")" == "forgotten $FORGOTTEN_ID" ]]
[[ "$(run_rb "$RUSTY_BRAIN_PROJECT" forget "$FORGOTTEN_ID")" == "no such memory: $FORGOTTEN_ID" ]]
run_rb "$RUSTY_BRAIN_PROJECT" recent --limit 10 > "$VERIFY_ROOT/recent-after.json"
run_rb "$RUSTY_BRAIN_PROJECT" recall 'soft deletion' --limit 10 > "$VERIFY_ROOT/recall-after.json"
run_rb "$RUSTY_BRAIN_PROJECT" stats > "$VERIFY_ROOT/stats.json"
uv run --no-project python -I - "$VERIFY_ROOT" "$FORGOTTEN_ID" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
forgotten = sys.argv[2]
def absent(rows):
    return all((row.get("memory") or row)["id"] != forgotten for row in rows)
before = json.load(open(root / "recent-before.json", encoding="utf-8"))
after = json.load(open(root / "recent-after.json", encoding="utf-8"))
recall = json.load(open(root / "recall-after.json", encoding="utf-8"))
assert not absent(before), "active-row control did not make recent absence fail"
assert absent(after) and absent(recall), "forgotten row remains in an active view"
stats = json.load(open(root / "stats.json", encoding="utf-8"))
assert stats["memories"] == 9 and stats["model"] == "BGESmallENV15Q" and stats["schema_version"] == 3
print("forget: recent/recall absence controls passed; active stats=9")
PY

OTHER_PROJECT="rb-local-other"
OTHER_ID="$(run_rb "$OTHER_PROJECT" remember 'Other project memory stays isolated' --tags isolation)"
run_rb "$OTHER_PROJECT" recent --limit 10 > "$VERIFY_ROOT/other-recent.json"
run_rb "$RUSTY_BRAIN_PROJECT" recent --limit 10 > "$VERIFY_ROOT/main-isolation.json"
run_rb "$RUSTY_BRAIN_PROJECT" path > "$VERIFY_ROOT/main-path.txt"
run_rb "$OTHER_PROJECT" path > "$VERIFY_ROOT/other-path.txt"
uv run --no-project python -I - "$VERIFY_ROOT" "$OTHER_ID" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
other = json.load(open(root / "other-recent.json", encoding="utf-8"))
main = json.load(open(root / "main-isolation.json", encoding="utf-8"))
assert [row["id"] for row in other] == [sys.argv[2]]
assert len(main) == 9 and all(row["id"] != sys.argv[2] for row in main)
assert (root / "main-path.txt").read_bytes() != (root / "other-path.txt").read_bytes()
print("project isolation: explicit projects and paths are disjoint")
PY

printf 'Running six fresh-process embedding operations with networking denied\n'
OFFLINE_A="$(sandbox_rb "$RUSTY_BRAIN_PROJECT" remember 'Offline sandbox alpha remains local' --tags offline)"
sandbox_rb "$RUSTY_BRAIN_PROJECT" recall 'Offline sandbox alpha remains local' --limit 5 > "$VERIFY_ROOT/offline-a.json"
sandbox_rb "$RUSTY_BRAIN_PROJECT" recall 'customer-specific database instances stay warm' --limit 5 > "$VERIFY_ROOT/offline-semantic.json"
OFFLINE_B="$(sandbox_rb "$RUSTY_BRAIN_PROJECT" remember 'Offline sandbox beta remains local' --tags offline)"
sandbox_rb "$RUSTY_BRAIN_PROJECT" recall 'Offline sandbox beta remains local' --limit 5 > "$VERIFY_ROOT/offline-b.json"
sandbox_rb "$RUSTY_BRAIN_PROJECT" recall 'write-once' --limit 5 > "$VERIFY_ROOT/offline-keyword.json"
uv run --no-project python -I - "$VERIFY_ROOT" "$OFFLINE_A" "$OFFLINE_B" "${IDS[0]}" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
def ids(name):
    return [hit["memory"]["id"] for hit in json.load(open(root / name, encoding="utf-8"))]
assert sys.argv[2] in ids("offline-a.json")
assert sys.argv[3] in ids("offline-b.json")
semantic = ids("offline-semantic.json")
assert sys.argv[4] in semantic[:3]
assert ids("offline-keyword.json")
print("offline fresh-process operations=6; semantic/keyword ranks remain correct")
PY
sandbox_rb "$RUSTY_BRAIN_PROJECT" forget "$OFFLINE_A" >/dev/null
sandbox_rb "$RUSTY_BRAIN_PROJECT" forget "$OFFLINE_B" >/dev/null
[[ "$(run_rb "$RUSTY_BRAIN_PROJECT" stats | uv run --no-project python -I -c 'import json,sys; print(json.load(sys.stdin)["memories"])')" == "9" ]]

EMPTY_HOME="$VERIFY_ROOT/empty-cache-home"
set +e
/usr/bin/sandbox-exec -f "$PROFILE" /usr/bin/env -i \
    HOME="$HOME" PATH="$RUNTIME_PATH" TMPDIR="${TMPDIR:-/tmp}" LANG=en_US.UTF-8 \
    RUSTY_BRAIN_HOME="$EMPTY_HOME" "$BIN" --project empty-cache remember 'must not be stored offline' \
    >"$VERIFY_ROOT/empty-cache.stdout" 2>"$VERIFY_ROOT/empty-cache.stderr"
EMPTY_STATUS=$?
set -e
[[ "$EMPTY_STATUS" -ne 0 && ! -s "$VERIFY_ROOT/empty-cache.stdout" ]]
grep -Eiq 'first run.*download|download.*first|first.*download' "$VERIFY_ROOT/empty-cache.stderr"
if [[ -e "$EMPTY_HOME/projects/empty-cache/brain.db" ]]; then
    printf 'empty-cache failure mutated the store by creating brain.db\n' >&2
    exit 1
fi
mkdir -p "$VERIFY_ROOT/planted-control/projects/empty-cache"
: > "$VERIFY_ROOT/planted-control/projects/empty-cache/brain.db"
if [[ ! -e "$VERIFY_ROOT/planted-control/projects/empty-cache/brain.db" ]]; then
    printf 'planted brain positive control was not detected\n' >&2
    exit 1
fi
printf 'empty-cache offline failure: exit=%s, first-download guidance, no brain write\n' "$EMPTY_STATUS"

printf 'Observing ready-cache and empty-cache connections through loopback\n'
PROBE_DIR="$VERIFY_ROOT/network-probe"
mkdir -p "$PROBE_DIR"
uv run --no-project python -I "$RB/scripts/network_probe.py" serve --state-dir "$PROBE_DIR" \
    >"$VERIFY_ROOT/probe.stdout" 2>"$VERIFY_ROOT/probe.stderr" &
PROBE_PID=$!
for _ in $(seq 1 100); do
    [[ -f "$PROBE_DIR/ready.json" ]] && break
    /bin/sleep 0.05
done
[[ -f "$PROBE_DIR/ready.json" ]]
PROBE_PORT="$(uv run --no-project python -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["port"])' "$PROBE_DIR/ready.json")"
ENDPOINT="http://127.0.0.1:$PROBE_PORT"
env -i HOME="$HOME" PATH="$RUNTIME_PATH" TMPDIR="${TMPDIR:-/tmp}" LANG=en_US.UTF-8 \
    HF_ENDPOINT="$ENDPOINT" RUSTY_BRAIN_HOME="$RUSTY_BRAIN_HOME" \
    "$BIN" --project "$RUSTY_BRAIN_PROJECT" recall 'write-once' --limit 1 > "$VERIFY_ROOT/probe-ready.json"
READY_SUMMARY="$(uv run --no-project python -I "$RB/scripts/network_probe.py" summary --state-dir "$PROBE_DIR")"
uv run --no-project python -I -c 'import json,sys; s=json.loads(sys.argv[1]); assert s["connections"] == 0 and s["requests"] == 0, s; print("ready-cache probe:", s)' "$READY_SUMMARY"

set +e
env -i HOME="$HOME" PATH="$RUNTIME_PATH" TMPDIR="${TMPDIR:-/tmp}" LANG=en_US.UTF-8 \
    HF_ENDPOINT="$ENDPOINT" RUSTY_BRAIN_HOME="$VERIFY_ROOT/probe-empty-home" \
    "$BIN" --project probe-empty remember 'loopback control must fail' \
    >"$VERIFY_ROOT/probe-empty.stdout" 2>"$VERIFY_ROOT/probe-empty.stderr"
PROBE_EMPTY_STATUS=$?
set -e
[[ "$PROBE_EMPTY_STATUS" -ne 0 ]]
grep -Eiq 'first run.*download|download.*first|first.*download' "$VERIFY_ROOT/probe-empty.stderr"
if [[ -e "$VERIFY_ROOT/probe-empty-home/projects/probe-empty/brain.db" ]]; then
    printf 'loopback empty-cache control created brain.db before model readiness\n' >&2
    exit 1
fi
EMPTY_SUMMARY="$(uv run --no-project python -I "$RB/scripts/network_probe.py" summary --state-dir "$PROBE_DIR")"
uv run --no-project python -I -c 'import json,sys; s=json.loads(sys.argv[1]); assert s["connections"] >= 1 and s["requests"] >= 1, s; print("empty-cache positive-control probe:", s)' "$EMPTY_SUMMARY"
kill "$PROBE_PID"
wait "$PROBE_PID" || true
PROBE_PID=""

printf 'Reopening a copy of the retained base-revision brain\n'
mkdir -p "$RUSTY_BRAIN_HOME/projects/base-fixture"
cp "$RB/tests/fixtures/base-f586d4e/brain.db" "$RUSTY_BRAIN_HOME/projects/base-fixture/brain.db"
BASE_SOURCE_HASH_BEFORE="$(shasum -a 256 "$RB/tests/fixtures/base-f586d4e/brain.db" | cut -d' ' -f1)"
run_rb base-fixture recent --limit 10 > "$VERIFY_ROOT/base-recent.json"
run_rb base-fixture stats > "$VERIFY_ROOT/base-stats.json"
uv run --no-project python -I - "$VERIFY_ROOT" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
rows = json.load(open(root / "base-recent.json", encoding="utf-8"))
stats = json.load(open(root / "base-stats.json", encoding="utf-8"))
assert len(rows) == 2 and stats["memories"] == 2
assert {row["content"] for row in rows} == {
    "Base fixture: DuckDB remains local",
    "Base fixture: Parquet preserves vectors",
}
print("base brain: two active rows readable; forgotten row excluded")
PY
BASE_SOURCE_HASH_AFTER="$(shasum -a 256 "$RB/tests/fixtures/base-f586d4e/brain.db" | cut -d' ' -f1)"
[[ "$BASE_SOURCE_HASH_BEFORE" == "$BASE_SOURCE_HASH_AFTER" ]]

printf 'Exporting populated, rapid-successive, zero-row, and base-revision snapshots\n'
MAIN_DB="$RUSTY_BRAIN_HOME/projects/$RUSTY_BRAIN_PROJECT/brain.db"
SOURCE_HASH_BEFORE="$(shasum -a 256 "$MAIN_DB" | cut -d' ' -f1)"
run_rb "$RUSTY_BRAIN_PROJECT" export > "$VERIFY_ROOT/export-1.stdout"
PARSED="$(uv run --no-project python -I -c 'import re,sys; m=re.fullmatch(r"exported (\d+) memories to (.+)", open(sys.argv[1]).read().strip()); assert m; print(m[2] + "|" + m[1])' "$VERIFY_ROOT/export-1.stdout")"
IFS='|' read -r EXPORT1_PATH EXPORT1_COUNT <<< "$PARSED"
[[ -f "$EXPORT1_PATH" ]]
uv run --isolated "$RB/scripts/verify_parquet.py" --db "$MAIN_DB" --parquet "$EXPORT1_PATH" --reported-count "$EXPORT1_COUNT"
EXPORT1_HASH="$(shasum -a 256 "$EXPORT1_PATH" | cut -d' ' -f1)"

run_rb "$RUSTY_BRAIN_PROJECT" export > "$VERIFY_ROOT/export-2.stdout"
PARSED="$(uv run --no-project python -I -c 'import re,sys; m=re.fullmatch(r"exported (\d+) memories to (.+)", open(sys.argv[1]).read().strip()); assert m; print(m[2] + "|" + m[1])' "$VERIFY_ROOT/export-2.stdout")"
IFS='|' read -r EXPORT2_PATH EXPORT2_COUNT <<< "$PARSED"
[[ "$EXPORT1_PATH" != "$EXPORT2_PATH" && -f "$EXPORT2_PATH" ]]
[[ "$(shasum -a 256 "$EXPORT1_PATH" | cut -d' ' -f1)" == "$EXPORT1_HASH" ]]
uv run --isolated "$RB/scripts/verify_parquet.py" --db "$MAIN_DB" --parquet "$EXPORT2_PATH" --reported-count "$EXPORT2_COUNT"
cp "$EXPORT1_PATH" "$VERIFY_ROOT/mutated-copy.parquet"
printf 'mutation' >> "$VERIFY_ROOT/mutated-copy.parquet"
if cmp -s "$EXPORT1_PATH" "$VERIFY_ROOT/mutated-copy.parquet"; then
    printf 'mutated-copy positive control was not detected\n' >&2
    exit 1
fi
printf 'rapid exports: distinct paths; prior hash preserved; mutated-copy control rejected\n'

EMPTY_PROJECT="rb-empty-export"
run_rb "$EMPTY_PROJECT" export > "$VERIFY_ROOT/export-empty.stdout"
PARSED="$(uv run --no-project python -I -c 'import re,sys; m=re.fullmatch(r"exported (\d+) memories to (.+)", open(sys.argv[1]).read().strip()); assert m; print(m[2] + "|" + m[1])' "$VERIFY_ROOT/export-empty.stdout")"
IFS='|' read -r EMPTY_EXPORT_PATH EMPTY_EXPORT_COUNT <<< "$PARSED"
[[ "$EMPTY_EXPORT_COUNT" == "0" ]]
EMPTY_DB="$RUSTY_BRAIN_HOME/projects/$EMPTY_PROJECT/brain.db"
uv run --isolated "$RB/scripts/verify_parquet.py" --db "$EMPTY_DB" --parquet "$EMPTY_EXPORT_PATH" --reported-count 0

BASE_PARQUET="$RB/tests/fixtures/base-f586d4e/memories.parquet"
BASE_DB="$RB/tests/fixtures/base-f586d4e/brain.db"
BASE_PARQUET_HASH="$(shasum -a 256 "$BASE_PARQUET" | cut -d' ' -f1)"
uv run --isolated "$RB/scripts/verify_parquet.py" --db "$BASE_DB" --parquet "$BASE_PARQUET" --reported-count 2
[[ "$(shasum -a 256 "$BASE_PARQUET" | cut -d' ' -f1)" == "$BASE_PARQUET_HASH" ]]
SOURCE_HASH_AFTER="$(shasum -a 256 "$MAIN_DB" | cut -d' ' -f1)"
[[ "$SOURCE_HASH_BEFORE" == "$SOURCE_HASH_AFTER" ]]

EXPORT_DIR="$RUSTY_BRAIN_HOME/projects/$RUSTY_BRAIN_PROJECT/exports"
snapshot_exports() {
    find "$EXPORT_DIR" -maxdepth 1 -type f -print | LC_ALL=C sort
}
printf 'Injecting write, close, hard-link, and post-link cleanup failures\n'
for STAGE in write close hard-link cleanup; do
    BEFORE_FILES="$(snapshot_exports)"
    set +e
    env -i HOME="$HOME" PATH="$RUNTIME_PATH" TMPDIR="${TMPDIR:-/tmp}" LANG=en_US.UTF-8 \
        RUSTY_BRAIN_HOME="$RUSTY_BRAIN_HOME" RUSTY_BRAIN_TEST_EXPORT_FAULT="$STAGE" \
        "$BIN" --project "$RUSTY_BRAIN_PROJECT" export \
        >"$VERIFY_ROOT/export-fault-$STAGE.stdout" 2>"$VERIFY_ROOT/export-fault-$STAGE.stderr"
    FAULT_STATUS=$?
    set -e
    if [[ "$FAULT_STATUS" -eq 0 ]]; then
        printf 'export %s fault unexpectedly succeeded\n' "$STAGE" >&2
        exit 1
    fi
    [[ ! -s "$VERIFY_ROOT/export-fault-$STAGE.stdout" ]]
    case "$STAGE" in
        write) grep -qi 'write' "$VERIFY_ROOT/export-fault-$STAGE.stderr" ;;
        close) grep -qi 'close' "$VERIFY_ROOT/export-fault-$STAGE.stderr" ;;
        hard-link) grep -Eqi 'hard.?link' "$VERIFY_ROOT/export-fault-$STAGE.stderr" ;;
        cleanup) grep -qi 'cleanup' "$VERIFY_ROOT/export-fault-$STAGE.stderr" ;;
    esac
    AFTER_FILES="$(snapshot_exports)"
    if [[ "$BEFORE_FILES" != "$AFTER_FILES" ]]; then
        printf 'export %s fault left a new final/temp artifact\n' "$STAGE" >&2
        exit 1
    fi
    [[ "$(shasum -a 256 "$EXPORT1_PATH" | cut -d' ' -f1)" == "$EXPORT1_HASH" ]]
    [[ "$(shasum -a 256 "$MAIN_DB" | cut -d' ' -f1)" == "$SOURCE_HASH_BEFORE" ]]
    printf 'export fault=%s exit=%s: actionable stderr, no success/final/temp, source/prior unchanged\n' "$STAGE" "$FAULT_STATUS"
done

BEFORE_FILES="$(snapshot_exports)"
: > "$EXPORT_DIR/memories-planted-final.parquet"
if [[ "$BEFORE_FILES" == "$(snapshot_exports)" ]]; then
    printf 'planted-final positive control was not detected\n' >&2
    exit 1
fi
rm "$EXPORT_DIR/memories-planted-final.parquet"
[[ "$BEFORE_FILES" == "$(snapshot_exports)" ]]
printf 'H2 confirmed: populated/zero/base schemas and all vectors verified through DuckDB 2.0.0\n'
printf 'H1 confirmed: six sandboxed fresh-process embedding operations; zero ready-cache attempts; empty-cache control observed\n'
printf 'CLI/store/export verification passed; evidence root: %s\n' "$VERIFY_ROOT"
