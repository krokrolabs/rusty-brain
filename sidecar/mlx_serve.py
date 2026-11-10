"""rusty-brain MLX embedding sidecar — MIT. Owns the ModelBERT-on-MLX encoder
(no mlx-embeddings dependency; mlx and transformers only, both commercially
licensed). Speaks line-delimited JSON on stdin/stdout; stdout is protocol-only,
diagnostics go to stderr. Loads local artifacts only — networking belongs to
`rusty-brain setup-mlx`.

Protocol (one JSON object per line):
  -> {"op":"load","artifact":"<dir>","model":"<id>"}   (first line after start)
  <- {"ok":true,"op":"load","model":..,"dim":..,"protocol":1,"versions":{..}}
  -> {"op":"embed","id":N,"text":"<prefix applied text>"}
  <- {"id":N,"embedding":[...]}                       (L2-normalized fp32)
  -> {"op":"shutdown"} -> exit 0
Errors: {"id":N,"error":"<actionable message>"} without the embedding.
The device is MLX's default (Metal when present, CPU otherwise).
"""
import json
import math
import re
import numpy as np
import sys
from pathlib import Path

import mlx.core as mx

PROTOCOL = 1
WARM_SHAPES = (32, 64, 128, 256, 512)


def log(message):
    print(f"[mlx-sidecar] {message}", file=sys.stderr, flush=True)


# --- ModernBERT on mlx.core (mirrors HF ModernBertModel; MIT, self-owned) ----

def _layer_norm(x, weight, eps):
    return mx.fast.layer_norm(x.astype(mx.float16), weight, None, eps)


def _rope(x, theta, dims):
    return mlx_nn.RoPE(dims=dims, base=theta)(x)


class _Dequantized:
    """Dequantized fp16 weight matrix, loaded from a packed artifact tensor."""

    def __init__(self):
        self.weight = None

    def load(self, weights, key):
        key = key[: -len(".weight")] if key.endswith(".weight") else key
        self.weight = mx.dequantize(
            weights[f"{key}.weight"], weights[f"{key}.scales"],
            weights[f"{key}.biases"], 64, 4).astype(mx.float16)

    def forward(self, x):
        return x.astype(mx.float16) @ self.weight.T

    def rows(self, ids):
        return self.weight[ids.reshape(-1).astype(mx.int32)].reshape(
            ids.shape[0], ids.shape[1], self.weight.shape[1])


class _Attention:
    def __init__(self, cfg, index):
        self.heads = cfg["num_attention_heads"]
        self.dim = cfg["hidden_size"] // self.heads
        self.global_layer = index % cfg["global_attn_every_n_layers"] == 0
        self.window = cfg["local_attention"] // 2  # HF half-window convention
        self.theta = float(cfg["global_rope_theta"] if self.global_layer
                           else cfg["local_rope_theta"])
        self.scale = 1.0 / math.sqrt(self.dim)
        self.Wqkv = _Dequantized()
        self.Wo = _Dequantized()

    def __call__(self, x, pad_mask):
        B, L = x.shape[0], x.shape[1]
        H, D = self.heads, self.dim
        qkv = self.Wqkv.forward(x).reshape((B, L, 3 * H, D)).transpose((0, 2, 1, 3))
        q, k, v = qkv[:, :H], qkv[:, H:2 * H], qkv[:, 2 * H:]
        rope = mlx_nn.RoPE(dims=D, base=self.theta)
        q, k = rope(q), rope(k)
        mask = pad_mask
        if not self.global_layer:
            pos = mx.arange(L)
            delta = pos[:, None] - pos[None, :]
            blocked = mx.where(mx.abs(delta) <= self.window, 0.0, -mx.inf)
            mask = mask + blocked[None, :, :].astype(x.dtype)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose((0, 2, 1, 3)).reshape((B, L, H * D))
        return self.Wo.forward(out)


class _Mlp:
    def __init__(self):
        self.Wi = _Dequantized()
        self.Wo = _Dequantized()

    def __call__(self, x):
        fused = self.Wi.forward(x)
        first, gate = mx.split(fused, 2, axis=-1)
        half = first.astype(mx.float32)
        gelu = ((mx.erf(half / math.sqrt(2.0)) * 0.5 + 0.5) * half).astype(mx.float16)
        return self.Wo.forward(gelu * gate)


class _Layer:
    def __init__(self, cfg, index):
        self.first = index == 0
        self.attn = _Attention(cfg, index)
        self.mlp = _Mlp()
        self.mlp_norm = mx.zeros((cfg["hidden_size"],), dtype=mx.float16)
        self.attn_norm = self.mlp_norm  # layer 0: Identity, overwritten on load
        self.eps = cfg.get("norm_eps", 1e-5)

    def __call__(self, x, mask):
        pre = x if self.first else _layer_norm(x, self.attn_norm, self.eps)
        x = x + self.attn(pre, mask)
        return x + self.mlp(_layer_norm(x, self.mlp_norm, self.eps))


class ModernBert:
    def __init__(self, artifact):
        self.artifact = Path(artifact)
        self.tokenizer = None
        self.tok = None
        self.layers = []
        self.dim = None

    def load(self):
        cfg = json.loads((self.artifact / "config.json").read_text())
        if cfg.get("model_type") != "modernbert":
            raise ValueError(
                f"unsupported model_type {cfg.get('model_type')!r}; "
                "the built-in MLX encoder is ModernBERT-only")
        weights = mx.load(str(self.artifact / "model.safetensors"))
        eps = cfg.get("norm_eps", 1e-5)
        self.dim = cfg["hidden_size"]
        self.tok = mx.dequantize(
            weights["model.embeddings.tok_embeddings.weight"],
            weights["model.embeddings.tok_embeddings.scales"],
            weights["model.embeddings.tok_embeddings.biases"], 64, 4
        ).astype(mx.float16)
        self.embeddings_norm = weights["model.embeddings.norm.weight"].astype(mx.float16)
        pattern = re.compile(
            r"model\.layers\.(\d+)\.(attn|mlp)\.(Wqkv|Wi|Wo)\.weight$")
        parts = {}
        for key in weights:
            found = pattern.match(key)
            if found:
                parts[(int(found.group(1)), found.group(2), found.group(3))] = key
        self.layers = []
        for index in range(cfg["num_hidden_layers"]):
            layer = _Layer(cfg, index)
            layer.attn.Wqkv.load(weights, parts[(index, "attn", "Wqkv")])
            layer.attn.Wo.load(weights, parts[(index, "attn", "Wo")])
            layer.mlp.Wi.load(weights, parts[(index, "mlp", "Wi")])
            layer.mlp.Wo.load(weights, parts[(index, "mlp", "Wo")])
            layer.mlp_norm = weights[
                f"model.layers.{index}.mlp_norm.weight"].astype(mx.float16)
            if index != 0:  # layer 0's attention norm is Identity in HF
                layer.attn_norm = weights[
                    f"model.layers.{index}.attn_norm.weight"].astype(mx.float16)
            self.layers.append(layer)
            mx.eval(layer.mlp.Wo.weight, layer.mlp_norm)
        self.final_norm = weights["model.final_norm.weight"].astype(mx.float16)
        mx.eval(self.tok, self.embeddings_norm, self.final_norm)

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.artifact))
        self._warmup()
        return {"dim": int(self.dim), "layers": int(cfg["num_hidden_layers"])}

    def _warmup(self):
        # Bucketed pads + 2 passes each: pre-compiles the Metal graph per shape
        # so no single call pays the compile tail (measured 131-171 ms without).
        rng = mx.random.key(0)
        for shape in WARM_SHAPES:
            ids = mx.zeros((1, shape), dtype=mx.int32)
            mask = mx.ones((1, shape), dtype=mx.float16)
            if shape >= 4:
                ids = mx.random.randint(1000, 50000, (1, shape), key=rng)
            for _ in range(2):
                mx.eval(self._encode(ids, mask))
        mx.eval(self._encode(mx.zeros((1, 1), dtype=mx.int32),
                             mx.ones((1, 1), dtype=mx.float16)))

    def _encode(self, ids, pad_mask):
        x = self.tok[ids.reshape(-1).astype(mx.int32)].reshape(
            ids.shape[0], ids.shape[1], self.dim)  # int64 index fine for gather
        x = _layer_norm(x, self.embeddings_norm, eps=1e-05)
        mask = mx.where(pad_mask[:, None, None, :].astype(mx.bool_), 0.0, -mx.inf)
        mask = mask.astype(x.dtype)
        for layer in self.layers:
            x = layer(x, mask)
        x = _layer_norm(x, self.final_norm, eps=1e-05)
        m = pad_mask[:, :, None].astype(x.dtype)
        pooled = (x * m).sum(axis=1) / mx.clip(m.sum(axis=1), 1.0, None)
        norm = mx.sqrt(mx.sum(pooled * pooled, axis=-1, keepdims=True))
        return (pooled / mx.clip(norm, 1e-12, None)).astype(mx.float32)

    def embed(self, text):
        encoded = self.tokenizer([text], truncation=True, max_length=512,
                                 padding=False, return_tensors="np")
        ids = mx.array(encoded["input_ids"])  # tokenizer np -> int64; cast via itemsize
        mask = mx.array(encoded["attention_mask"])  # int64 -> later fp16 in _encode
        return self._encode(ids, mask)[0].tolist()


def main():
    loaded = None
    versions = {"mlx": mx.__version__, "protocol": PROTOCOL}
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as error:
            log(f"unparsable request: {error}")
            continue
        op = request.get("op")
        try:
            if op == "load":
                if loaded is not None:
                    raise ValueError("sidecar already loaded a model; restart to switch")
                modernbert = ModernBert(request["artifact"])
                info = modernbert.load()
                versions["dim"] = info["dim"]
                loaded = modernbert
                print(json.dumps({"ok": True, "op": "load", "protocol": PROTOCOL,
                                  "model": request.get("model"), "dim": info["dim"],
                                  "versions": versions}), flush=True)
            elif op == "embed":
                if loaded is None:
                    raise ValueError("model not loaded yet")
                embedding = loaded.embed(request["text"])
                print(json.dumps({"id": request.get("id"), "embedding": embedding}),
                      flush=True)
            elif op == "shutdown":
                print(json.dumps({"ok": True, "op": "shutdown"}), flush=True)
                return 0
            else:
                print(json.dumps({"error": f"unknown op: {op!r}"}), flush=True)
        except Exception as error:  # noqa: BLE001 - sidecar stays alive
            payload = {"id": request.get("id"), "error": str(error)}
            print(json.dumps(payload), flush=True)
            log(f"error for op {op!r}: {error}")
    return 0


import mlx.nn as mlx_nn  # noqa: E402 - used by _rope/_Layer

if __name__ == "__main__":
    if not mx.metal.is_available():
        log("Metal unavailable; running on CPU (slower but correct).")
    try:
        mx.set_default_device(mx.gpu)
    except Exception:  # noqa: BLE001
        log("GPU unavailable; using CPU default device.")
    sys.exit(main())
