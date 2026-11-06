#!/usr/bin/env python3
"""Static contract checks for the public installation and verification docs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import sys
import tempfile


TROUBLESHOOTING = {
    "Missing or incomplete model cache": "model cache",
    "Unavailable dsh, Ollama, or local model": "dsh/Ollama/model",
    "Missing rusty-brain executable": "executable",
    "Invalid mount, config, cwd, or override": "mount/config/cwd/override",
    "Failed Parquet or DuckDB readback": "Parquet/DuckDB",
    "Failed network-denial positive control": "network-denial control",
}
FIELDS = ("Observable symptom", "Likely cause", "Corrective action")

README_REQUIRED = (
    "cargo install --path .",
    "rusty-brain remember",
    "rusty-brain recall",
    "rusty-brain stats",
    "rusty-brain path",
    "rusty-brain forget",
    "rusty-brain export",
    "rusty-brain serve",
    "rusty-brain reembed",
    "rusty-brain models",
    "scripts/verify-local.sh",
    "scripts/verify-dsh.sh",
    "cargo test --offline",
    "BGESmallENV15Q",
    "SnowflakeArcticEmbedMQ",
    "claude mcp add rusty-brain",
    "RUSTY_BRAIN_AGENT",
    "RUSTY_BRAIN_SESSION",
    "memory_embeddings", "--prune",
    "DuckDB",
    "macOS arm64",
    "--agent self",
    "docs/parquet-schema.md",
    "docs/mcp-compatibility.md",
    "docs/verification.md",
)

VERIFICATION_REQUIRED = (
    "f586d4e6683842085165d808df8a5021f2f67210",
    "Current base SHA",
    "Commands and observed results",
    "cargo fmt --check",
    "cargo clippy --all-targets --all-features -- -D warnings",
    "cargo test --offline",
    "cargo build --offline",
    "scripts/verify-local.sh",
    "scripts/verify-dsh.sh",
    "uv run --isolated scripts/verify_docs.py --self-test",
    "RUSTY_BRAIN_E2E=1 RUSTY_BRAIN_HOME=\"$RUSTY_BRAIN_HOME\"",
    "H1",
    "H2",
    "H3",
    "Defects fixed",
    "Initial live evaluation",
    "Limitations",
    "final tree OID",
    "dirty disclosure",
    "post-commit SHA",
    "detached",
)

FIXTURE_ROWS = (
    ("A", "DuckDB instances are reused per customer", "decision,duckdb"),
    ("B", "Parquet exports are derived, write-once snapshots", "principle,parquet"),
    ("C", "The local brain stores data under the project directory", "storage"),
    ("D", "PostgreSQL connection pooling is not used by rusty-brain", "negative"),
    ("E", "The MCP server communicates over standard input and output", "mcp"),
    ("F", "Forgetting a memory is a soft deletion", "lifecycle"),
    ("G", "The embedding model produces 384 values", "embedding"),
    ("H", "Use uv to run the DuckDB verification script", "verification"),
    ("I", "Ollama serves the local agent model", "ollama"),
    ("J", "deepseek-harness mounts memory through a patch overlay", "dsh"),
)

LINK_RE = re.compile(r"(?<!!)\[[^]]+\]\(([^)]+)\)")
HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*$")
FIELD_RE = re.compile(r"^[-*]\s+\*\*(Observable symptom|Likely cause|Corrective action):\*\*\s*(.*)$")


@dataclass
class Documents:
    root: Path
    readme: str
    verification: str


def load_documents(root: Path) -> Documents:
    return Documents(
        root=root,
        readme=(root / "README.md").read_text(encoding="utf-8"),
        verification=(root / "docs/verification.md").read_text(encoding="utf-8"),
    )


def check_links(root: Path, relative: str, text: str) -> list[str]:
    failures = []
    source = root / relative
    for match in LINK_RE.finditer(text):
        target = match.group(1).strip().split()[0].strip("<>")
        if not target or target.startswith(("#", "http://", "https://", "mailto:")):
            continue
        path_text = target.split("#", 1)[0]
        if not path_text:
            continue
        resolved = (source.parent / path_text).resolve()
        try:
            resolved.relative_to(root.resolve())
        except ValueError:
            failures.append(f"{relative}: link escapes documentation root: {target}")
            continue
        if not resolved.exists():
            failures.append(f"{relative}: missing local link target: {target}")
    return failures


def troubleshooting_sections(readme: str) -> dict[str, str]:
    lines = readme.splitlines()
    sections: dict[str, str] = {}
    for index, line in enumerate(lines):
        heading = HEADING_RE.match(line)
        if not heading or not line.startswith("### "):
            continue
        title = heading.group(1)
        normalized = re.sub(r"^\d+[.)]\s+", "", title)
        if normalized not in TROUBLESHOOTING:
            continue
        end = len(lines)
        for candidate in range(index + 1, len(lines)):
            if lines[candidate].startswith("### ") or lines[candidate].startswith("## "):
                end = candidate
                break
        sections[normalized] = "\n".join(lines[index + 1 : end])
    return sections


def check_troubleshooting(readme: str) -> list[str]:
    failures = []
    sections = troubleshooting_sections(readme)
    for category, label in TROUBLESHOOTING.items():
        body = sections.get(category)
        if body is None:
            failures.append(f"README.md: missing troubleshooting category: {category}")
            continue
        present = {match.group(1): match.group(2).strip() for match in map(FIELD_RE.match, body.splitlines()) if match}
        for field in FIELDS:
            if not present.get(field):
                failures.append(f"README.md: {category}: missing/nonempty {field}")
    return failures


def check_documents(documents: Documents) -> list[str]:
    failures: list[str] = []
    for diagram, block in enumerate(re.findall(r"```mermaid\s*\n(.*?)```", documents.readme, re.DOTALL), start=1):
        for label in re.findall(r"(?:-->|-.->|==>)\|([^|\n]*)\|", block):
            label = label.strip()
            if re.search(r"[{\[]", label) and not (label.startswith('"') and label.endswith('"')):
                failures.append(f"README.md: mermaid diagram {diagram}: unquoted brace/bracket in edge label")
    for snippet in README_REQUIRED:
        if snippet not in documents.readme:
            failures.append(f"README.md: missing required text/command: {snippet}")
    for snippet in VERIFICATION_REQUIRED:
        if snippet not in documents.verification:
            failures.append(f"docs/verification.md: missing required text/command: {snippet}")
    for key, content, tags in FIXTURE_ROWS:
        row = f"| {key} | `{content}` | `{tags}` |"
        if row not in documents.verification:
            failures.append(f"docs/verification.md: missing exact fixture row {key}")
    failures.extend(check_links(documents.root, "README.md", documents.readme))
    failures.extend(check_links(documents.root, "docs/verification.md", documents.verification))
    failures.extend(check_troubleshooting(documents.readme))
    if documents.readme.count("```mermaid") < 3:
        failures.append("README.md: expected at least 3 mermaid diagrams")

    forbidden_evidence = (
        r"(?im)^\s*(?:[-*]\s*)?(?:\*\*)?final tree OID(?:\*\*)?\s*:\s*[0-9a-f]{40}\b",
        r"(?im)^\s*(?:[-*]\s*)?(?:\*\*)?post-commit SHA(?:\*\*)?\s*:\s*[0-9a-f]{40}\b",
        r"(?im)^\s*(?:[-*]\s*)?(?:\*\*)?dirty disclosure(?:\*\*)?\s*:\s*(?:clean|modified|dirty|[0-9a-f]{40})",
    )
    for pattern in forbidden_evidence:
        if re.search(pattern, documents.verification):
            failures.append("docs/verification.md: detached tree/dirty/commit evidence leaked into tracked docs")
    return failures


def replace_once(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, f"positive-control anchor count for {old!r}: {text.count(old)}"
    return text.replace(old, new, 1)


def run_self_test(root: Path) -> list[str]:
    failures: list[str] = []
    documents = load_documents(root)
    with tempfile.TemporaryDirectory(prefix="rusty-brain-docs-controls.") as temporary:
        control_root = Path(temporary)
        shutil.copy2(root / "README.md", control_root / "README.md")
        (control_root / "docs").mkdir()
        shutil.copy2(root / "docs/verification.md", control_root / "docs/verification.md")
        for relative in ("docs/parquet-schema.md", "docs/mcp-compatibility.md"):
            shutil.copy2(root / relative, control_root / relative)
        (control_root / "contracts").mkdir()
        shutil.copy2(root / "contracts/mcp-tools.json", control_root / "contracts/mcp-tools.json")
        fixture = control_root / "tests/fixtures/base-f586d4e"
        fixture.mkdir(parents=True)
        for name in ("brain.db", "memories.parquet", "README.md"):
            shutil.copy2(root / "tests/fixtures/base-f586d4e" / name, fixture / name)

        mutated = replace_once(documents.readme, "cargo test --offline", "cargo test --offline # removed-command-control")
        # Remove the command entirely, including this occurrence's commentary.
        mutated = mutated.replace("cargo test --offline # removed-command-control", "offline test command removed", 1)
        (control_root / "README.md").write_text(mutated, encoding="utf-8")
        control_failures = check_documents(load_documents(control_root))
        if not any("cargo test --offline" in failure for failure in control_failures):
            failures.append("positive control: removing cargo fmt --check was not rejected")
        else:
            print("positive control rejected: required command removal")

        for category in TROUBLESHOOTING:
            for field in FIELDS:
                shutil.copy2(root / "README.md", control_root / "README.md")
                original = load_documents(control_root).readme
                body = troubleshooting_sections(original)[category]
                match = next(
                    match for match in map(FIELD_RE.match, body.splitlines())
                    if match and match.group(1) == field
                )
                mutated = replace_once(original, match.group(0), f"- **{field}:**")
                (control_root / "README.md").write_text(mutated, encoding="utf-8")
                control_failures = check_documents(load_documents(control_root))
                expected = f"{category}: missing/nonempty {field}"
                if not any(expected in failure for failure in control_failures):
                    failures.append(f"positive control not rejected: {expected}")
                else:
                    print(f"positive control rejected: {category} / {field}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--self-test", action="store_true", help="also require all removal controls to fail")
    args = parser.parse_args()
    try:
        documents = load_documents(args.root)
    except OSError as error:
        print(f"documentation check setup failed: {error}", file=sys.stderr)
        return 2
    failures = check_documents(documents)
    if not failures and args.self_test:
        failures.extend(run_self_test(args.root))
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        print(f"documentation verification failed: {len(failures)} finding(s)", file=sys.stderr)
        return 1
    suffix = " and 19 removal controls" if args.self_test else ""
    print(f"documentation verification passed: links, topics, six troubleshooting categories{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
