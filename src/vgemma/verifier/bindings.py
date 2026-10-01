"""Receipt-level bindings: schema, signature, roots, manifest, seed, prompt and IO chain."""

from __future__ import annotations

from typing import Any

from vgemma.profile import GemmaProfile
from vgemma.protocol import (
    LM_HEAD_ID,
    RECEIPT_VERSION,
    SAMPLER_ID,
    SOFTCAP_ID,
    SUPPORTED_ATTN,
    check_receipt_signature,
    client_seed,
    io_chain,
    prompt_hash,
    seed_commitment,
)
from vgemma.verifier.report import Ledger, VerifyFail

_HEX64 = set("0123456789abcdef")

RECEIPT_FIELDS: dict[str, type | tuple[type, ...]] = {
    "version": int,
    "request_id": str,
    "model": dict,
    "manifest": dict,
    "prompt_hash": str,
    "seed_commitment": str,
    "n_prompt": int,
    "n_gen": int,
    "trace_root": str,
    "io_chain_head": str,
    "prover": dict,
}
MANIFEST_FIELDS: dict[str, type | tuple[type, ...]] = {
    "dtype": str,
    "attn_implementation": str,
    "temperature": (int, float),
    "top_p": (int, float),
    "top_k": int,
    "greedy": bool,
    "final_logit_softcapping": (int, float),
    "embed_scale_bf16": (int, float),
    "rms_norm_eps": (int, float),
    "sliding_window": int,
    "layer_types_hash": str,
    "attention_k_eq_v": bool,
    "rope_hash": str,
    "qk_norm": bool,
    "thinking": bool,
    "chat_template_hash": str,
    "tokenizer_hash": str,
    "speculative": str,
    "prefix_caching": bool,
    "max_new_tokens": int,
    "eos_token_ids": list,
    "sampler": str,
    "softcap_impl": str,
    "lm_head": str,
}
POLICY_FIELDS = ("temperature", "top_k", "top_p", "greedy", "max_new_tokens", "thinking")
EXPECTED_MANIFEST_FIELDS = (*POLICY_FIELDS, "attn_implementation", "tokenizer_hash", "chat_template_hash")


def _typed(v: Any, t: type | tuple[type, ...]) -> bool:
    """isinstance, except that a bool never passes for a number."""
    return isinstance(v, t) and (t is bool or not isinstance(v, bool))


def _is_hash(s: Any) -> bool:
    return isinstance(s, str) and len(s) == 64 and set(s) <= _HEX64


def check_receipt_schema(receipt: Any, ledger: Ledger) -> None:
    def ok(cond: bool, msg: str) -> None:
        ledger.exact("RECEIPT_SCHEMA", cond, msg)

    ok(isinstance(receipt, dict), "receipt is not a JSON object")
    ok(receipt.get("version") == RECEIPT_VERSION, f"receipt version {receipt.get('version')!r} != {RECEIPT_VERSION}")
    for k, t in RECEIPT_FIELDS.items():
        ok(_typed(receipt.get(k), t), f"field {k} missing or not {t}")
    for k in ("prompt_hash", "seed_commitment", "trace_root", "io_chain_head"):
        ok(_is_hash(receipt[k]), f"{k} is not a 32-byte hex digest")
    for k in ("weights_root", "config_hash"):
        ok(_is_hash(receipt["model"].get(k)), f"model.{k} is not a 32-byte hex digest")
    nonce = receipt.get("client_nonce")
    ok(nonce is None or _is_hash(nonce), "client_nonce must be null or a 32-byte hex value")
    m = receipt["manifest"]
    for k, t in MANIFEST_FIELDS.items():
        ok(_typed(m.get(k), t), f"manifest.{k} missing or not {t}")
    ok(receipt["n_prompt"] >= 1 and receipt["n_gen"] >= 1, "token counts must be >= 1")


def check_signature(receipt: dict[str, Any], ledger: Ledger, prover_id: str | None) -> None:
    good, why = check_receipt_signature(receipt)
    ledger.exact("RECEIPT_SIGNATURE", good, why)
    if prover_id is not None:
        ledger.exact(
            "RECEIPT_SIGNATURE",
            receipt["prover"]["id"] == prover_id,
            f"prover id is not the pinned {prover_id[:24]}...",
        )


def check_roots(receipt: dict[str, Any], public: dict[str, Any], key_meta: dict[str, Any], ledger: Ledger) -> None:
    ledger.exact(
        "WEIGHTS_ROOT", public["weights_root"] == key_meta["weights_root"], "public params do not match the key"
    )
    ledger.exact(
        "WEIGHTS_ROOT",
        receipt["model"]["weights_root"] == public["weights_root"],
        f"receipt weights_root {receipt['model']['weights_root'][:16]}... != public {public['weights_root'][:16]}...",
    )
    ledger.exact("CONFIG_HASH", public["config_hash"] == key_meta["config_hash"], "public params do not match the key")
    ledger.exact(
        "WEIGHTS_ROOT",
        public["embedding_root"] == key_meta["embedding_root"],
        "public embedding root does not match the key",
    )
    ledger.exact(
        "CONFIG_HASH",
        receipt["model"]["config_hash"] == public["config_hash"],
        "receipt config_hash != public config_hash",
    )


def check_manifest(m: dict[str, Any], profile: GemmaProfile, public: dict[str, Any], ledger: Ledger) -> None:
    unsupported = []
    if m["speculative"] != "none":
        unsupported.append(f"speculative={m['speculative']}")
    if m["prefix_caching"]:
        unsupported.append("prefix_caching")
    if m["attn_implementation"] not in SUPPORTED_ATTN:
        unsupported.append(f"attn_implementation={m['attn_implementation']}")
    if m["dtype"] != "bfloat16":
        unsupported.append(f"dtype={m['dtype']}")
    if m["sampler"] != SAMPLER_ID or m["softcap_impl"] != SOFTCAP_ID or m["lm_head"] != LM_HEAD_ID:
        unsupported.append(f"sampler/softcap/lm_head implementation {m['sampler']}/{m['softcap_impl']}/{m['lm_head']}")
    if not m["greedy"] and not (m["temperature"] > 0 and 0 < m["top_p"] <= 1 and m["top_k"] >= 0):
        unsupported.append("sampling parameters out of range")
    if m["max_new_tokens"] < 1:
        unsupported.append("max_new_tokens < 1")
    ledger.exact("MANIFEST_UNSUPPORTED", not unsupported, "; ".join(unsupported))

    mism = []
    for k, want in (
        ("final_logit_softcapping", profile.final_logit_softcapping),
        ("embed_scale_bf16", profile.embed_scale_bf16),
        ("rms_norm_eps", profile.rms_norm_eps),
    ):
        if float(m[k]) != float(want):
            mism.append(f"{k}={m[k]} (checkpoint {want})")
    for k in ("eos_token_ids", "tokenizer_hash", "chat_template_hash"):
        if k in public and (sorted(m[k]) if k == "eos_token_ids" else m[k]) != public[k]:
            mism.append(f"{k} differs from the public params")
    ledger.exact("MANIFEST_MISMATCH", not mism, "; ".join(mism))

    wiring = []
    for k, want in (
        ("sliding_window", profile.sliding_window),
        ("layer_types_hash", profile.layer_types_hash()),
        ("attention_k_eq_v", profile.attention_k_eq_v),
        ("rope_hash", profile.rope_hash()),
        ("qk_norm", True),
    ):
        if m[k] != want:
            wiring.append(f"manifest {k}={m[k]!r}, profile {want!r}")
    ledger.exact("WIRING", not wiring, "; ".join(wiring))


def check_expected(receipt: dict[str, Any], expected: dict[str, Any], ledger: Ledger) -> None:
    """What the client actually asked for, against what the prover's manifest declares."""
    m = receipt["manifest"]
    diff = [
        f"{k}: manifest {m[k]!r}, requested {expected[k]!r}"
        for k in EXPECTED_MANIFEST_FIELDS
        if k in expected and m[k] != expected[k]
    ]
    ledger.exact("MANIFEST_MISMATCH", not diff, "; ".join(diff))
    if "client_nonce" in expected:
        ledger.exact(
            "SEED_COMMITMENT",
            receipt.get("client_nonce") == expected["client_nonce"],
            "receipt does not carry the client's nonce",
        )


def check_seed(receipt: dict[str, Any], seed_hex: Any, ledger: Ledger) -> bytes:
    ok = isinstance(seed_hex, str) and _is_hash(seed_hex)
    ledger.exact("SEED_COMMITMENT", ok, "revealed seed missing or malformed")
    seed = bytes.fromhex(seed_hex)
    if receipt.get("client_nonce"):
        ledger.exact(
            "SEED_COMMITMENT",
            seed == client_seed(bytes.fromhex(receipt["client_nonce"])),
            "seed is not H(client nonce): the prover chose its own randomness",
        )
    ledger.exact(
        "SEED_COMMITMENT",
        seed_commitment(seed, receipt["request_id"]).hex() == receipt["seed_commitment"],
        "H(seed || request_id) != seed_commitment",
    )
    return seed


def check_prompt(receipt: dict[str, Any], prompt_tokens: Any, expected: dict[str, Any], ledger: Ledger) -> list[int]:
    ok = isinstance(prompt_tokens, list) and all(
        isinstance(t, int) and not isinstance(t, bool) and t >= 0 for t in prompt_tokens
    )
    ledger.exact("PROMPT_BINDING", ok, "prompt tokens missing or malformed (the opening must reveal the prompt)")
    ledger.exact(
        "PROMPT_BINDING",
        len(prompt_tokens) == receipt["n_prompt"],
        f"{len(prompt_tokens)} prompt tokens opened, receipt says n_prompt={receipt['n_prompt']}",
    )
    ledger.exact("PROMPT_BINDING", prompt_hash(prompt_tokens).hex() == receipt["prompt_hash"], "prompt hash mismatch")
    if expected:
        ledger.exact(
            "PROMPT_BINDING",
            "prompt_tokens" in expected or "prompt_hash" in expected,
            "the client request carries no prompt binding (template the prompt client-side)",
        )
    if "prompt_tokens" in expected:
        ledger.exact("PROMPT_BINDING", prompt_tokens == list(expected["prompt_tokens"]), "not the client's prompt")
    if "prompt_hash" in expected:
        ledger.exact("PROMPT_BINDING", receipt["prompt_hash"] == expected["prompt_hash"], "not the client's prompt")
    return prompt_tokens


def check_io_chain(receipt: dict[str, Any], transcript: Any, ledger: Ledger) -> list[tuple[int, bytes]]:
    try:
        tr = [(int(t), bytes.fromhex(h)) for t, h in transcript]
    except (TypeError, ValueError) as e:
        raise VerifyFail("IO_CHAIN", f"io transcript malformed: {e}") from e
    ledger.exact(
        "IO_CHAIN", len(tr) == receipt["n_gen"], f"transcript has {len(tr)} tokens, receipt n_gen={receipt['n_gen']}"
    )
    head = io_chain(bytes.fromhex(receipt["prompt_hash"]), tr)
    ledger.exact("IO_CHAIN", head.hex() == receipt["io_chain_head"], "IO chain head mismatch")
    m = receipt["manifest"]
    eos = set(m["eos_token_ids"])
    early = [i for i, (t, _) in enumerate(tr[:-1]) if t in eos]
    ledger.exact("IO_CHAIN", not early, f"EOS token at step {early[:1]} before the end of the transcript")
    ledger.exact(
        "IO_CHAIN",
        tr[-1][0] in eos or len(tr) == m["max_new_tokens"],
        f"generation stopped after {len(tr)} tokens without EOS (max_new_tokens={m['max_new_tokens']})",
    )
    ledger.exact("IO_CHAIN", len(tr) <= m["max_new_tokens"], "more tokens than max_new_tokens")
    return tr
