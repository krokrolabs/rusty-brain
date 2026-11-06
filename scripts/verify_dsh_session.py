#!/usr/bin/env python3
"""Independently verify one isolated dsh JSONL session and its brain database."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
from typing import Any
import uuid


EXPECTED = [
    "mcp__rusty-brain__remember",
    "mcp__rusty-brain__recall",
    "mcp__rusty-brain__stats",
    "mcp__rusty-brain__data_path",
    "mcp__rusty-brain__forget",
    "mcp__rusty-brain__recall",
]


def message_text(message: Any) -> str:
    if not isinstance(message, dict):
        return ""
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content
            if isinstance(part, dict) and part.get("type") in (None, "text")
        )
    return ""


def tool_message(result: dict[str, Any]) -> dict[str, Any]:
    message = result.get("data", {}).get("message", {})
    content = message.get("content", []) if isinstance(message, dict) else []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool-result":
                return block
    return message if isinstance(message, dict) else {}


def result_payload(result: dict[str, Any]) -> dict[str, Any]:
    message = tool_message(result)
    structured = message.get("structuredContent")
    if isinstance(structured, dict):
        return structured
    content = message.get("content", [])
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "text":
                continue
            try:
                parsed = json.loads(block.get("text", ""))
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list):
                return {"hits": parsed, "truncated": False}
    raise ValueError("tool result has no structuredContent or JSON text fallback")


def recall_hits(payload: dict[str, Any]) -> list[dict[str, Any]]:
    hits = payload.get("hits")
    if not isinstance(hits, list):
        raise ValueError("recall result has no hits array")
    return [hit for hit in hits if isinstance(hit, dict)]


def usage_totals(events: list[dict[str, Any]]) -> dict[str, Any]:
    usages = [
        event["data"]["usage"] for event in events
        if event.get("type") == "assistant/message"
        and isinstance(event.get("data"), dict)
        and isinstance(event["data"].get("usage"), dict)
    ]
    if not usages:
        return {"available": False}
    totals: dict[str, int] = {}
    for usage in usages:
        for key, value in usage.items():
            if isinstance(value, int) and not isinstance(value, bool):
                totals[key] = totals.get(key, 0) + value
    for key in ("inputTokens", "outputTokens"):
        totals.setdefault(key, 0)
    total_source = "sum of exposed totalTokens"
    if "totalTokens" not in totals:
        totals["totalTokens"] = totals["inputTokens"] + totals["outputTokens"]
        total_source = "inputTokens + outputTokens; totalTokens was not separately exposed"
    return {"available": True, "messages": len(usages), "totals": totals, "totalTokensSource": total_source}


def evaluate(root: Path, token: str, task: str) -> dict[str, Any]:
    sessions = sorted((root / "sessions").rglob("session*.jsonl"))
    if len(sessions) != 1:
        raise ValueError(f"expected exactly one new isolated session*.jsonl, found {len(sessions)}")
    session = sessions[0]
    events = [json.loads(line) for line in session.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not events:
        raise ValueError("session JSONL is empty")
    project_cwd = str((root / "work/project").resolve())
    failures: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    # dsh 0.1.1-rc.2 emits header cwd at top level; accept meta.cwd from the
    # reviewed field contract as well, but never skip the resolved-cwd check.
    header_cwd = events[0].get("meta", {}).get("cwd") or events[0].get("cwd")
    check(
        isinstance(header_cwd, str) and Path(header_cwd).resolve() == Path(project_cwd).resolve(),
        "session header meta.cwd differs from temporary project cwd",
    )
    calls = [event for event in events[1:] if event.get("type") == "tool/call"]
    results = [event for event in events[1:] if event.get("type") == "tool/result"]
    names = [call.get("data", {}).get("name") for call in calls]
    check(names == EXPECTED, f"public tool sequence mismatch: {names!r}")
    counts = Counter(name if isinstance(name, str) else "<missing>" for name in names)

    by_call: dict[str, dict[str, Any]] = {}
    for result in results:
        data = result.get("data", {})
        message = tool_message(result)
        call_id = message.get("toolCallId") or data.get("message", {}).get("source", {}).get("callId")
        if isinstance(call_id, str):
            by_call[call_id] = result

    arguments: list[dict[str, Any]] = []
    for index, call in enumerate(calls):
        data = call.get("data", {})
        raw = data.get("arguments")
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else None
        except json.JSONDecodeError:
            parsed = None
        if not isinstance(parsed, dict):
            failures.append(f"call {index + 1} arguments are not a JSON-encoded object")
            parsed = {}
        arguments.append(parsed)
        check(isinstance(data.get("callId"), str), f"call {index + 1} lacks callId")
        check("turn" in data and "step" in data, f"call {index + 1} lacks turn/step")
        result = by_call.get(data.get("callId", ""))
        if result is None:
            failures.append(f"call {index + 1} has no joined tool/result")
        else:
            message = tool_message(result)
            check(message.get("isError") is not True, f"call {index + 1} returned isError")
            check(result.get("data", {}).get("error") is None, f"call {index + 1} returned data.error")

    memory_id = ""
    before_hits: list[dict[str, Any]] = []
    after_hits: list[dict[str, Any]] = []
    path_payload: dict[str, Any] = {}
    stats_payload: dict[str, Any] = {}
    if len(calls) >= 6 and len(arguments) >= 6 and all(call.get("data", {}).get("callId") in by_call for call in calls[:6]):
        try:
            remember = result_payload(by_call[calls[0]["data"]["callId"]])
            memory_id = remember.get("id", "")
            parsed_id = uuid.UUID(memory_id)
            check(str(parsed_id) == memory_id, "remember did not return a canonical UUID")
            check(arguments[0] == {"content": f"Verification token: {token}", "tags": "verification"}, "remember arguments mismatch")

            before_payload = result_payload(by_call[calls[1]["data"]["callId"]])
            before_hits = recall_hits(before_payload)
            check(arguments[1] == {"query": token, "limit": 5}, "first recall arguments mismatch")
            present = any(hit.get("memory", {}).get("id") == memory_id and token in hit.get("memory", {}).get("content", "") for hit in before_hits)
            check(present, "first recall did not return the remembered token/id")
            absent_before = all(token not in hit.get("memory", {}).get("content", "") for hit in before_hits)
            check(not absent_before, "active-row positive control did not make forgotten-absence predicate fail")

            stats_payload = result_payload(by_call[calls[2]["data"]["callId"]])
            check(arguments[2] == {}, "stats arguments must be empty")
            check(stats_payload.get("model") == "AllMiniLML6V2Q", "stats model mismatch")
            check(isinstance(stats_payload.get("memories"), int) and stats_payload["memories"] >= 1, "stats did not include active memory")

            path_payload = result_payload(by_call[calls[3]["data"]["callId"]])
            check(arguments[3] == {}, "data_path arguments must be empty")
            expected_db = root / "home/projects/rb-dsh-e2e/brain.db"
            check(path_payload.get("home") == str(root / "home"), "data_path home did not resolve isolated root")
            check(path_payload.get("project") == "rb-dsh-e2e", "data_path project override mismatch")
            check(path_payload.get("brain_file") == str(expected_db), "data_path brain path mismatch")

            check(arguments[4] == {"id": memory_id}, "forget did not use remember's exact UUID")
            forgotten = result_payload(by_call[calls[4]["data"]["callId"]])
            check(forgotten == {"id": memory_id, "forgotten": True}, "forget result mismatch")

            after_payload = result_payload(by_call[calls[5]["data"]["callId"]])
            after_hits = recall_hits(after_payload)
            check(arguments[5] == {"query": token, "limit": 5}, "absence recall arguments mismatch")
            check(all(token not in hit.get("memory", {}).get("content", "") for hit in after_hits), "final recall still contains token")
        except (KeyError, ValueError, TypeError) as error:
            failures.append(f"tool payload validation failed: {error}")

    active_count = -1
    db_path = root / "home/projects/rb-dsh-e2e/brain.db"
    try:
        uri = f"file:{db_path}?mode=ro"
        with sqlite3.connect(uri, uri=True) as connection:
            row = connection.execute(
                "SELECT content, deleted_at FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
            check(row is not None, "independent DB has no remembered source row")
            if row is not None:
                check(row[0] == f"Verification token: {token}", "independent DB content mismatch")
                check(row[1] is not None, "independent DB source row is not soft-deleted")
            active_count = connection.execute("SELECT count(*) FROM memories WHERE deleted_at IS NULL").fetchone()[0]
            check(connection.execute("SELECT count(*) FROM memories WHERE content LIKE ? AND deleted_at IS NULL", (f"%{token}%",)).fetchone()[0] == 0, "active DB still contains token")
            check(connection.execute("SELECT count(*) FROM memories_fts WHERE id = ?", (memory_id,)).fetchone()[0] == 0, "forgotten memory remains in FTS")
    except (sqlite3.Error, OSError) as error:
        failures.append(f"independent DB inspection failed: {error}")

    assistant = []
    for event in events[1:]:
        if event.get("type") != "assistant/message" or not isinstance(event.get("data"), dict):
            continue
        message = event["data"].get("message")
        interrupted = event["data"].get("interrupted") is True or (
            isinstance(message, dict) and (
                message.get("interrupted") is True or message.get("stopReason") == "interrupted"
            )
        )
        if not interrupted and message_text(message):
            assistant.append(message_text(message))
    final = assistant[-1] if assistant else ""
    check(token in final, "final response lacks live token")
    check(memory_id and memory_id in final, "final response lacks remembered UUID")
    check(str(active_count) in final, "final response lacks independently observed active count")
    check("verified absent" in final.lower(), "final response does not confirm verified absent")

    return {
        "scenario": "remember-recall-stats-data_path-forget-absence",
        "pass": 0 if failures else 1,
        "model": "qwen3.8:27b-mlx",
        "client": "dsh 0.1.1-rc.2",
        "contractRevision": "initial-verified-v1",
        "session": str(session),
        "memoryId": memory_id,
        "activeCountAfterForget": active_count,
        "toolCalls": {"total": len(calls), "perTool": dict(sorted(counts.items()))},
        "tokenUsage": usage_totals(events[1:]),
        "task": task,
        "failures": failures,
        "baseline": "initial one-scenario CT-5A baseline; no prior-baseline comparison",
    }


def failure_record(root: Path, task: str, error: str) -> dict[str, Any]:
    return {
        "scenario": "remember-recall-stats-data_path-forget-absence",
        "pass": 0, "model": "qwen3.8:27b-mlx", "client": "dsh 0.1.1-rc.2",
        "contractRevision": "initial-verified-v1", "toolCalls": {"total": 0, "perTool": {}},
        "tokenUsage": {"available": False}, "task": task, "failures": [error],
        "sessionRoot": str(root / "sessions"),
        "baseline": "initial one-scenario CT-5A baseline; no prior-baseline comparison",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--task-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    task = args.task_file.read_text(encoding="utf-8")
    try:
        record = evaluate(args.root.absolute(), args.token, task)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        record = failure_record(args.root, task, str(error))
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, sort_keys=True))
    return 0 if record["pass"] == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
