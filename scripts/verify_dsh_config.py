#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pyyaml==6.0.3"]
# ///
"""Safely check exact dsh overlays and literal, unevaluated dump-config fields."""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import sys
from typing import Any

import yaml


PROJECT = "rb-dsh-e2e"
SESSION_EXPRESSION = "process.env.RUSTY_BRAIN_VERIFY_ROOT + '/sessions'"
ARGS_EXPRESSION = (
    "process.env.RUSTY_BRAIN_PROJECT\n"
    "  ? ['--project', process.env.RUSTY_BRAIN_PROJECT, 'serve']\n"
    "  : ['serve']"
)
CWD_EXPRESSION = "process.cwd()"
MEMORY_ENV = {"RUSTY_BRAIN_AGENT": "dsh", "RUSTY_BRAIN_SESSION": "dsh-headless"}


class Loader(yaml.SafeLoader):
    """SafeLoader that preserves JavaScript source as an inert string."""


Loader.add_constructor("!!js", lambda loader, node: loader.construct_scalar(node))
Loader.add_constructor("tag:yaml.org,2002:js", lambda loader, node: loader.construct_scalar(node))


def load(path: Path) -> Any:
    value = yaml.load(path.read_text(encoding="utf-8"), Loader=Loader)
    reject_nulls(value, str(path))
    return value


def reject_nulls(value: Any, location: str) -> None:
    if value is None:
        raise ValueError(f"null-valued key at {location}")
    if isinstance(value, dict):
        for key, child in value.items():
            reject_nulls(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_nulls(child, f"{location}[{index}]")


def by_id(document: Any, plugin_id: str) -> dict[str, Any]:
    if not isinstance(document, list):
        raise ValueError("composed dsh config must be a list of plugin entries")
    matches = [entry for entry in document if isinstance(entry, dict) and entry.get("id") == plugin_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {plugin_id} entry, found {len(matches)}")
    return matches[0]


def inserted_memory(overlay: Any) -> dict[str, Any]:
    if not isinstance(overlay, list) or len(overlay) != 1:
        raise ValueError("memory overlay must contain one insert patch")
    inserted = overlay[0].get("insert")
    if not isinstance(inserted, list) or len(inserted) != 1:
        raise ValueError("memory overlay insert must contain one MCP client")
    return inserted[0]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate(document: Any, root: Path, project_cwd: Path, env: dict[str, str]) -> None:
    require(root.is_absolute() and root.name.startswith("rusty-brain-verify."), "verification root is not isolated")
    require(project_cwd.is_absolute() and project_cwd.is_relative_to(root), "project cwd escapes verification root")
    require(env.get("RUSTY_BRAIN_VERIFY_ROOT") == str(root), "RUSTY_BRAIN_VERIFY_ROOT override mismatch")
    require(env.get("RUSTY_BRAIN_HOME") == str(root / "home"), "RUSTY_BRAIN_HOME override mismatch")
    require(env.get("RUSTY_BRAIN_PROJECT") == PROJECT, "RUSTY_BRAIN_PROJECT override mismatch")

    require(by_id(document, "llm-deepseek").get("disabled") is True, "hosted DeepSeek provider is enabled")
    require(by_id(document, "web-search-deepseek").get("disabled") is True, "hosted web search is enabled")
    require(by_id(document, "tool-web").get("disabled") is True, "web tool is enabled")
    require(by_id(document, "session-telemetry-otel").get("config", {}).get("mode") == "DISABLED", "telemetry is not disabled")

    ollama = by_id(document, "llm-pi-ai").get("config", {}).get("providers", {}).get("ollama", {})
    require(ollama.get("apiKeyEnv") == "RB_TEST_SECRET", "Ollama must use only the generated local test marker")
    require(ollama.get("api") == "openai-completions", "Ollama API mismatch")
    require(ollama.get("baseURL") == "http://127.0.0.1:11434/v1", "Ollama must use loopback")
    models = ollama.get("models")
    require(models == [{"id": "qwen3.8:27b-mlx", "contextWindow": 262144, "maxTokens": 4096}], "pinned Ollama model config drift")
    default = by_id(document, "agent-default-model").get("config", {})
    require(default.get("provider") == "ollama" and default.get("model") == "qwen3.8:27b-mlx", "default model is not pinned local Ollama")

    session = by_id(document, "session-persistence-jsonl").get("config", {})
    require(session.get("root") == SESSION_EXPRESSION, "session root !!js expression was evaluated or changed")
    require(session.get("compression") == "none", "sessions must be uncompressed JSONL")

    memory = by_id(document, "memory-rusty-brain")
    require(memory.get("name") == "@deepseek-ai/dsh-mcp-client", "wrong MCP client plugin")
    config = memory.get("config", {})
    require(config.get("serverName") == "rusty-brain", "serverName identity mismatch")
    require(config.get("transport") == "stdio", "memory transport must be stdio")
    require(config.get("command") == "rusty-brain", "memory command mismatch")
    require(config.get("env") == MEMORY_ENV, "dsh agent/session environment mismatch")
    require(config.get("args") == ARGS_EXPRESSION, "project args !!js expression was evaluated or changed")
    require(config.get("cwd") == CWD_EXPRESSION, "cwd !!js expression was evaluated or changed")


def run_positive_controls(document: Any, root: Path, project_cwd: Path, env: dict[str, str]) -> None:
    hosted = copy.deepcopy(document)
    by_id(hosted, "llm-deepseek")["disabled"] = False
    try:
        validate(hosted, root, project_cwd, env)
    except ValueError:
        pass
    else:
        raise AssertionError("hosted-provider positive control was accepted")

    null_control = copy.deepcopy(document)
    by_id(null_control, "agent-default-model")["config"]["provider"] = None
    try:
        reject_nulls(null_control, "control")
    except ValueError:
        pass
    else:
        raise AssertionError("null-valued positive control was accepted")

    wrong_identity = copy.deepcopy(document)
    by_id(wrong_identity, "memory-rusty-brain")["config"]["env"]["RUSTY_BRAIN_AGENT"] = "cli"
    try:
        validate(wrong_identity, root, project_cwd, env)
    except ValueError:
        pass
    else:
        raise AssertionError("agent-identity positive control was accepted")

    try:
        validate(document, root, Path("/tmp"), env)
    except ValueError:
        pass
    else:
        raise AssertionError("outside-root cwd positive control was accepted")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--project-cwd", type=Path, required=True)
    args = parser.parse_args()
    try:
        root = args.root.absolute()
        project_cwd = args.project_cwd.absolute()
        require(root.is_dir() and project_cwd.is_dir(), "verification root/project must exist")
        ollama_overlay = load(args.repo / "examples/deepseek-harness/ollama-local.cordis.yml")
        memory_overlay = load(args.repo / "examples/deepseek-harness/rusty-brain.cordis.yml")
        expected_ollama = [
            {"id": "llm-deepseek", "disabled": True},
            {"id": "llm-pi-ai", "config": {"providers": {"ollama": {
                "apiKeyEnv": "RB_TEST_SECRET", "api": "openai-completions", "baseURL": "http://127.0.0.1:11434/v1",
                "models": [{"id": "qwen3.8:27b-mlx", "contextWindow": 262144, "maxTokens": 4096}],
            }}}},
            {"id": "agent-default-model", "config": {"provider": "ollama", "model": "qwen3.8:27b-mlx"}},
            {"id": "session-telemetry-otel", "config": {"mode": "DISABLED"}},
            {"id": "web-search-deepseek", "disabled": True},
            {"id": "tool-web", "disabled": True},
            {"id": "session-persistence-jsonl", "config": {"root": SESSION_EXPRESSION, "compression": "none"}},
        ]
        require(ollama_overlay == expected_ollama, "Ollama overlay differs from the interoperability contract")
        memory = inserted_memory(memory_overlay)
        require(memory == {
            "id": "memory-rusty-brain", "name": "@deepseek-ai/dsh-mcp-client",
            "config": {"serverName": "rusty-brain", "transport": "stdio", "command": "rusty-brain", "env": MEMORY_ENV, "args": ARGS_EXPRESSION, "cwd": CWD_EXPRESSION},
        }, "memory overlay differs from the interoperability contract")
        document = load(args.dump)
        env = dict(os.environ)
        validate(document, root, project_cwd, env)
        run_positive_controls(document, root, project_cwd, env)
    except (OSError, ValueError, AssertionError, yaml.YAMLError) as error:
        print(f"dsh config check failed: {error}", file=sys.stderr)
        return 1
    print("dsh config: exact overlays, literal !!js, local-only policy, null/hosted/path controls passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
