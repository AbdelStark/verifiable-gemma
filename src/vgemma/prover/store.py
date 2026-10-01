"""Retained state: per-request trace tensors and commitment data on disk, with a TTL."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vgemma.protocol import decode_step


@dataclass
class Trace:
    """Everything the prover captured for one request (bf16 tensors as uint16 bits)."""

    request_id: str
    tokens: list[int]  # prompt then generated
    n_prompt: int
    n_gen: int
    layers: dict[int, dict[str, np.ndarray]]  # layer (num_layers = final norm group) -> name -> [F, ...]
    logits_precap: np.ndarray  # [n_gen, vocab] f32
    logits_postcap: np.ndarray  # [n_gen, vocab] f32
    witnesses: list[dict[str, Any]]
    seed: bytes
    client_nonce: bytes | None = None

    @property
    def n_positions(self) -> int:
        """Forward positions: the last sampled token is never fed back through the model."""
        return self.n_prompt + self.n_gen - 1

    def decode_step(self, pos: int) -> int | None:
        return decode_step(self.n_prompt, self.n_gen, pos)


@dataclass
class Commitment:
    receipt: dict[str, Any]
    layer_leaves: np.ndarray  # [F, num_layers + 1, 32] uint8
    position_bodies: np.ndarray  # [F, 32] uint8
    position_leaves: np.ndarray  # [F, 32] uint8
    logits_hashes: list[bytes]
    witness_hashes: list[bytes]


class RetainedStore:
    def __init__(self, root: Path, ttl_seconds: float = 3600.0, max_requests: int = 64):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl_seconds
        self.max_requests = max_requests

    def _dir(self, request_id: str) -> Path:
        if not request_id.replace("_", "").isalnum():
            raise ValueError(f"bad request id {request_id!r}")
        return self.root / request_id

    def save(self, trace: Trace, com: Commitment) -> int:
        from safetensors.numpy import save_file

        d = self._dir(trace.request_id)
        d.mkdir(parents=True, exist_ok=True)
        tensors = {f"trace/{layer}/{name}": arr for layer, g in trace.layers.items() for name, arr in g.items()}
        tensors["logits_precap"] = trace.logits_precap
        tensors["logits_postcap"] = trace.logits_postcap
        save_file(tensors, str(d / "trace.safetensors"))
        np.save(d / "layer_leaves.npy", com.layer_leaves)
        np.save(d / "position_bodies.npy", com.position_bodies)
        np.save(d / "position_leaves.npy", com.position_leaves)
        meta = {
            "request_id": trace.request_id,
            "tokens": trace.tokens,
            "n_prompt": trace.n_prompt,
            "n_gen": trace.n_gen,
            "witnesses": trace.witnesses,
            "seed": trace.seed.hex(),
            "client_nonce": trace.client_nonce.hex() if trace.client_nonce else None,
            "logits_hashes": [h.hex() for h in com.logits_hashes],
            "witness_hashes": [h.hex() for h in com.witness_hashes],
            "receipt": com.receipt,
            "created": time.time(),
        }
        (d / "meta.json").write_text(json.dumps(meta))
        self.evict()
        return sum(f.stat().st_size for f in d.iterdir())

    def load(self, request_id: str) -> tuple[Trace, Commitment]:
        from safetensors.numpy import load_file

        d = self._dir(request_id)
        if not (d / "meta.json").exists():
            raise KeyError(f"no retained state for {request_id} (expired or unknown)")
        meta = json.loads((d / "meta.json").read_text())
        raw = load_file(str(d / "trace.safetensors"))
        layers: dict[int, dict[str, np.ndarray]] = {}
        for k, v in raw.items():
            if k.startswith("trace/"):
                _, layer, name = k.split("/")
                layers.setdefault(int(layer), {})[name] = v
        trace = Trace(
            request_id=meta["request_id"],
            tokens=meta["tokens"],
            n_prompt=meta["n_prompt"],
            n_gen=meta["n_gen"],
            layers=layers,
            logits_precap=raw["logits_precap"],
            logits_postcap=raw["logits_postcap"],
            witnesses=meta["witnesses"],
            seed=bytes.fromhex(meta["seed"]),
            client_nonce=bytes.fromhex(meta["client_nonce"]) if meta.get("client_nonce") else None,
        )
        com = Commitment(
            receipt=meta["receipt"],
            layer_leaves=np.load(d / "layer_leaves.npy"),
            position_bodies=np.load(d / "position_bodies.npy"),
            position_leaves=np.load(d / "position_leaves.npy"),
            logits_hashes=[bytes.fromhex(h) for h in meta["logits_hashes"]],
            witness_hashes=[bytes.fromhex(h) for h in meta["witness_hashes"]],
        )
        return trace, com

    def evict(self) -> None:
        now = time.time()
        entries = []
        for d in self.root.iterdir():
            m = d / "meta.json"
            if d.is_dir() and m.exists():
                entries.append((m.stat().st_mtime, d))
        entries.sort()
        for i, (mtime, d) in enumerate(entries):
            if now - mtime > self.ttl or i < len(entries) - self.max_requests:
                shutil.rmtree(d, ignore_errors=True)

    def count(self) -> int:
        return sum(1 for d in self.root.iterdir() if (d / "meta.json").exists())
