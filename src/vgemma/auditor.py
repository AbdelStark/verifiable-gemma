"""Auditor side: choosing challenges (after the receipt is committed) and talking to the prover.

Challenged positions are drawn with the auditor's own randomness after the receipt is committed:
a prover that knew in advance which positions would be opened could cheat on all the others.
"""

from __future__ import annotations

import secrets
from pathlib import Path
from typing import Any

import httpx

from vgemma.protocol import decode_opening, normalize_audits


def routine_k(n_layers: int) -> int:
    return max(2, min(10, n_layers // 2))


def choose_layers(spec: str, n_layers: int) -> tuple[list[int], str]:
    """``full`` | ``routine`` | ``routine:k`` (uniform random subset, auditor randomness) | ``0,3,5``."""
    spec = spec.strip()
    if spec == "full":
        return list(range(n_layers)), "full"
    if spec.startswith("routine"):
        k = int(spec.split(":", 1)[1]) if ":" in spec else routine_k(n_layers)
        k = min(k, n_layers)
        rng = secrets.SystemRandom()
        return sorted(rng.sample(range(n_layers), k)), f"routine:{k}"
    layers = sorted({int(x) for x in spec.split(",") if x.strip()})
    return layers, "custom"


def decode_positions(receipt: dict[str, Any]) -> list[int]:
    """Forward positions whose logits produced a generated token."""
    n_prompt, n_gen = receipt["n_prompt"], receipt["n_gen"]
    return list(range(n_prompt - 1, n_prompt + n_gen - 1))


def choose_positions(spec: str, receipt: dict[str, Any]) -> list[int]:
    """``random`` / ``random:k`` (default 3 generated positions, secret randomness) | ``all-gen`` |
    ``none`` | ``edges`` (first, middle, last generated: predictable, for reproducible tests only) |
    ``gen:i,gen:j`` | ``17,98``."""
    dec = decode_positions(receipt)
    spec = spec.strip()
    if spec == "none":
        return []
    if spec == "edges":
        return sorted({dec[0], dec[len(dec) // 2], dec[-1]})
    if spec.startswith("random"):
        k = int(spec.split(":", 1)[1]) if ":" in spec else 3
        return sorted(secrets.SystemRandom().sample(dec, min(k, len(dec))))
    if spec == "all-gen":
        return dec
    out = []
    for part in spec.split(","):
        part = part.strip()
        if part.startswith("gen:"):
            out.append(dec[int(part[4:])])
        elif part:
            out.append(int(part))
    return sorted(set(out))


def make_challenge(
    receipt: dict[str, Any],
    n_layers: int,
    positions: str = "random",
    layers: str = "routine",
    attention: bool = True,
    decode: str = "all-gen",
    decode_layers: int = 1,
    decode_attention: bool = False,
) -> dict[str, Any]:
    """Per-position audits drawn with the auditor's randomness after the receipt is committed.

    ``positions`` get a full audit: their own independently drawn ``layers`` subset plus the
    attention audit. Every other ``decode`` position (by default every generated token) gets the
    decode checks and ``decode_layers`` independently drawn layers audited in full, so a residual
    stream forged at one layer boundary at every token is caught with probability that grows with
    the number of tokens: it escapes with ``prod (1 - |layers| / L)`` over all audits. A fake
    attention output is only caught by audits that replay attention; ``decode_attention`` extends the
    replay to the decode audits (costly on long contexts: every attended K/V row is opened).
    """
    rng = secrets.SystemRandom()
    audits, tier = [], "custom"
    full = choose_positions(positions, receipt)
    for p in full:
        layer_list, tier = choose_layers(layers, n_layers)
        audits.append({"pos": p, "layers": layer_list, "attention": attention})
    k = min(decode_layers, n_layers)
    for p in sorted(set(choose_positions(decode, receipt)) - set(full)):
        audits.append({"pos": p, "layers": sorted(rng.sample(range(n_layers), k)), "attention": decode_attention})
    return {"request_id": receipt["request_id"], "audits": sorted(audits, key=lambda a: a["pos"]), "tier": tier}


def challenge_from(
    request_id: str,
    positions: list[int],
    layers: list[int],
    attention: bool = True,
    decode_positions: list[int] = (),
    decode_layers: list[int] = (),
) -> dict[str, Any]:
    """A fixed challenge (tests, reproducible audits): the same layers at every full position."""
    audits = [{"pos": p, "layers": sorted(layers), "attention": attention} for p in sorted(set(positions))]
    audits += [
        {"pos": p, "layers": sorted(decode_layers), "attention": False}
        for p in sorted(set(decode_positions) - set(positions))
    ]
    return {"request_id": request_id, "audits": sorted(audits, key=lambda a: a["pos"]), "tier": "custom"}


def forged_boundary_escape(challenge: dict[str, Any], n_layers: int, attention_only: bool = False) -> float:
    """Prior probability that a prover forging one layer at every position escapes: a residual
    stream boundary (any audit of the producing layer catches it) or, with ``attention_only``, a fake
    attention output (only audits that replay attention catch it)."""
    out = 1.0
    for a in normalize_audits(challenge):
        if a["attention"] or not attention_only:
            out *= 1.0 - len(a["layers"]) / n_layers
    return out


def client_request(
    prompt: str,
    max_new_tokens: int,
    tokenizer_dir: Path,
    thinking: bool = False,
    greedy: bool = False,
    temperature: float = 1.0,
    top_k: int = 64,
    top_p: float = 0.95,
    prover_id: str | None = None,
    attn_implementation: str | None = None,
) -> dict[str, Any]:
    """What the client sends, with a fresh nonce that fixes the sampling randomness, recorded so the
    verifier can check the receipt answers this request (and not one the prover chose). The client
    templates the prompt itself with the checkpoint's public tokenizer."""
    messages = [{"role": "user", "content": prompt}]
    params = {
        "max_new_tokens": int(max_new_tokens),
        "thinking": bool(thinking),
        "greedy": bool(greedy),
        "temperature": float(temperature),
        "top_k": int(top_k),
        "top_p": float(top_p),
        "nonce": secrets.token_hex(32),
    }
    from vgemma.model import eos_token_ids, load_profile
    from vgemma.tokenizer import load_tokenizer

    _, cfg = load_profile(Path(tokenizer_dir))
    tok = load_tokenizer(Path(tokenizer_dir), eos_token_ids(Path(tokenizer_dir), cfg))
    return {
        "messages": messages,
        "params": params,
        "prover_id": prover_id,
        "prompt_tokens": tok.apply_chat(messages, thinking=thinking),
        "tokenizer_hash": tok.tokenizer_hash,
        "chat_template_hash": tok.chat_template_hash,
        "attn_implementation": attn_implementation,
    }


def expected_from_request(req: dict[str, Any]) -> dict[str, Any]:
    p = req["params"]
    exp = {k: p[k] for k in ("temperature", "top_k", "top_p", "greedy", "max_new_tokens", "thinking")}
    exp["client_nonce"] = p["nonce"]
    exp["prompt_tokens"] = req["prompt_tokens"]
    for k in ("tokenizer_hash", "chat_template_hash", "attn_implementation"):
        if req.get(k) is not None:
            exp[k] = req[k]
    return exp


class ProverClient:
    def __init__(self, base_url: str, timeout: float = 600.0):
        self.http = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def health(self) -> dict[str, Any]:
        r = self.http.get("/health")
        r.raise_for_status()
        return r.json()

    def chat(self, prompt: str, max_new_tokens: int = 64, **kw) -> dict[str, Any]:
        """``kw`` may carry sampling parameters, ``thinking`` and ``nonce`` (32-byte hex)."""
        r = self.http.post(
            "/chat", json={"messages": [{"role": "user", "content": prompt}], "max_new_tokens": max_new_tokens, **kw}
        )
        r.raise_for_status()
        return r.json()

    def audit(self, challenge: dict[str, Any]) -> bytes:
        r = self.http.post("/audit", json=challenge)
        r.raise_for_status()
        data = r.content
        echo = decode_opening(data)[0]["challenge"]
        if normalize_audits(echo) != normalize_audits(challenge):
            raise RuntimeError("prover answered a different challenge than the one sent")
        return data

    def bench(self, prompt: str, max_new_tokens: int, runs: int = 3) -> dict[str, Any]:
        r = self.http.post("/bench", json={"prompt": prompt, "max_new_tokens": max_new_tokens, "runs": runs})
        r.raise_for_status()
        return r.json()
