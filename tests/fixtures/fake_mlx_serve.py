"""Deterministic fake of the mlx_serve.py sidecar protocol — tests only. MIT.

Speaks the same line-delimited JSON contract as sidecar/mlx_serve.py without
importing mlx, transformers, or numpy: embeddings are sha256-derived so the
Rust test can expect them exactly. A LOAD_ERROR file placed next to the model
artifact directory makes the load reply fail (positive control for sidecar
error surfacing; per-test, no environment race).
"""

import hashlib
import json
import os
import sys

DIM = 768


def embed(text):
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    raw = [(digest[i % 32] - 128.0) / 128.0 for i in range(DIM)]
    norm = sum(v * v for v in raw) ** 0.5 or 1.0
    return [v / norm for v in raw]


def reply(payload):
    print(json.dumps(payload), flush=True)


for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    request = json.loads(line)
    op = request.get("op")
    if op == "load":
        marker = os.path.join(str(request.get("artifact", "")), "..", "LOAD_ERROR")
        if os.path.exists(marker):
            reply({"ok": False, "op": "load", "error": "fake sidecar load failure"})
        else:
            reply({"ok": True, "op": "load", "model": request.get("model"),
                   "dim": DIM, "protocol": 1, "versions": {}})
    elif op == "embed":
        if str(request.get("text", "")).endswith("__fail__"):
            reply({"id": request.get("id"), "error": "fake embed failure"})
        else:
            reply({"id": request.get("id"), "embedding": embed(request.get("text", ""))})
    elif op == "shutdown":
        reply({"ok": True, "op": "shutdown"})
        break
