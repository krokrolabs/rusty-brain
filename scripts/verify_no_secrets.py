#!/usr/bin/env python3
"""Fail if controlled evidence contains the generated marker or debug chains."""

from __future__ import annotations

import os
from pathlib import Path
import sys


def main() -> int:
    marker = os.environ.get("RB_TEST_SECRET") or os.environ.get("HF_TOKEN")
    if not marker:
        print("RB_TEST_SECRET or HF_TOKEN must contain a generated test marker", file=sys.stderr)
        return 2
    if len(marker) < 16:
        print("test marker must be at least 16 characters", file=sys.stderr)
        return 2
    if not sys.argv[1:]:
        print("usage: verify_no_secrets.py FILE [FILE ...]", file=sys.stderr)
        return 2

    failures: list[str] = []
    for argument in sys.argv[1:]:
        path = Path(argument)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as error:
            failures.append(f"{path}: unreadable ({error.strerror or 'I/O error'})")
            continue
        if marker in text:
            failures.append(f"{path}: generated marker credential leaked")
        if "Caused by:" in text:
            failures.append(f"{path}: diagnostic chain leaked")
        if "Stack backtrace" in text or "stack backtrace:" in text.lower():
            failures.append(f"{path}: stack trace leaked")

    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1
    print(f"secret/diagnostic scan passed for {len(sys.argv) - 1} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
