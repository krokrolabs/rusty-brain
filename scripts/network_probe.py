#!/usr/bin/env python3
"""Loopback-only HTTP/socket observer for local embedding cache verification."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import sys


def append_line(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def serve(state_dir: Path) -> int:
    state_dir.mkdir(parents=True, exist_ok=True)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(16)
    listener.settimeout(0.25)
    port = listener.getsockname()[1]
    ready = state_dir / "ready.json"
    ready.write_text(json.dumps({"host": "127.0.0.1", "pid": os.getpid(), "port": port}))
    running = True

    def stop(_signum: int, _frame: object) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while running:
            try:
                connection, _address = listener.accept()
            except TimeoutError:
                continue
            with connection:
                append_line(state_dir / "connections.log", "accepted")
                connection.settimeout(2.0)
                try:
                    request = connection.recv(16_384)
                except (TimeoutError, OSError):
                    request = b""
                if request:
                    # Persist only the request line; headers can carry credentials.
                    first_line = request.split(b"\r\n", 1)[0].decode("ascii", "replace")
                    append_line(state_dir / "requests.log", first_line)
                try:
                    body = b"deliberate loopback cache-control refusal\n"
                    connection.sendall(
                        b"HTTP/1.1 503 Service Unavailable\r\n"
                        + f"Content-Length: {len(body)}\r\n".encode("ascii")
                        + b"Connection: close\r\nContent-Type: text/plain\r\n\r\n"
                        + body
                    )
                except OSError:
                    pass
    finally:
        listener.close()
    return 0


def count_lines(path: Path) -> int:
    try:
        return sum(1 for _ in path.open("r", encoding="utf-8"))
    except FileNotFoundError:
        return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["serve", "summary"])
    parser.add_argument("--state-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "serve":
        return serve(args.state_dir)
    connections = count_lines(args.state_dir / "connections.log")
    requests = count_lines(args.state_dir / "requests.log")
    print(json.dumps({"connections": connections, "requests": requests}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
