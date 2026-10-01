"""Commitment formats shared by prover and verifier (TECH_SPEC section 6, docs/SCHEMAS.md).

Everything that is hashed is defined once here: layer and position leaves, the IO chain, seed
derivation and commitment, prompt hash, weight and embedding leaves, receipt hashing and
signing, and the binary opening container.
"""

from __future__ import annotations

import copy
import json
import struct
from collections.abc import Iterable
from typing import Any

import numpy as np

from vgemma.canon import EMPTY, H, canonical_json, i32, json_hash, tensor_hash, tensor_parts, u32
from vgemma.merkle import MerkleTree

RECEIPT_VERSION = 1
OPENING_VERSION = 1
OPENING_MAGIC = b"VGOPEN01"
SAMPLER_ID = "vg-sampler-1"
SOFTCAP_ID = "vg-canon-softcap-1"
LM_HEAD_ID = "f32"  # logits = f32(E) @ f32(h_final), no bf16 rounding of the output
SUPPORTED_ATTN = ("sdpa", "eager")

# ---------------------------------------------------------------------------
# Trace leaves
# ---------------------------------------------------------------------------


def layer_leaf(layer: int, pos: int, tensor_hashes: Iterable[bytes]) -> bytes:
    """``H("vg/layer" || layer || pos || H(t) for each captured name in profile order)``."""
    return H("vg/layer", u32(layer), u32(pos), *tensor_hashes)


def witness_hash(witness: dict[str, Any]) -> bytes:
    return json_hash("vg/witness", witness)


def position_body(
    layer_leaves: Iterable[bytes], logits_hash: bytes | None, out_token: int | None, witness_h: bytes | None
) -> bytes:
    """``H("vg/pos_body" || layer leaves (incl. final group) || H(logits_precap) || sampled token ||
    H(witness))``; prompt positions that sample nothing use EMPTY and -1."""
    return H(
        "vg/pos_body",
        *layer_leaves,
        logits_hash or EMPTY,
        i32(-1 if out_token is None else out_token),
        witness_h or EMPTY,
    )


def position_leaf(pos: int, input_token: int, body: bytes) -> bytes:
    """``H("vg/pos" || pos || input token || body)``: two levels, so an opening can bind the input
    token of every position for 36 bytes plus its proof, without opening anything else."""
    return H("vg/pos", u32(pos), u32(input_token), body)


# ---------------------------------------------------------------------------
# Bindings
# ---------------------------------------------------------------------------


def prompt_hash(tokens: list[int]) -> bytes:
    return H("vg/prompt", u32(len(tokens)), np.asarray(tokens, dtype="<u4").tobytes())


def io_init(p_hash: bytes) -> bytes:
    return H("vg/io", p_hash)


def io_step(prev: bytes, token: int, logits_hash: bytes) -> bytes:
    return H("vg/io", prev, u32(token), logits_hash)


def io_chain(p_hash: bytes, transcript: Iterable[tuple[int, bytes]]) -> bytes:
    c = io_init(p_hash)
    for token, lh in transcript:
        c = io_step(c, token, lh)
    return c


def derive_seed(prover_secret: bytes, request_id: str) -> bytes:
    return H("vg/seed", prover_secret, request_id.encode())


def client_seed(client_nonce: bytes) -> bytes:
    """Sampling seed fixed by a client nonce: the prover has no freedom left to grind over."""
    return H("vg/seed_client", client_nonce)


def seed_commitment(seed: bytes, request_id: str) -> bytes:
    return H("vg/seed_commit", seed, request_id.encode())


def logits_hash(logits_precap: np.ndarray) -> bytes:
    return tensor_hash(np.asarray(logits_precap, dtype=np.float32))


# ---------------------------------------------------------------------------
# Weights and embedding commitments (keygen and prover)
# ---------------------------------------------------------------------------


def weight_leaf(name: str, arr: np.ndarray) -> bytes:
    n = name.encode()
    return H("vg/weight", u32(len(n)), n, *tensor_parts(arr))


def weights_root(named_tensors: Iterable[tuple[str, np.ndarray]]) -> tuple[bytes, int]:
    leaves = [weight_leaf(n, a) for n, a in named_tensors]
    return MerkleTree(leaves).root, len(leaves)


def embedding_leaves(embed_bits: np.ndarray) -> list[bytes]:
    return [tensor_hash(embed_bits[i]) for i in range(embed_bits.shape[0])]


# ---------------------------------------------------------------------------
# Receipts
# ---------------------------------------------------------------------------


def receipt_signing_payload(receipt: dict[str, Any]) -> dict[str, Any]:
    r = copy.deepcopy(receipt)
    r.get("prover", {}).pop("signature", None)
    return r


def receipt_hash(receipt: dict[str, Any]) -> bytes:
    return H("vg/receipt", canonical_json(receipt_signing_payload(receipt)).encode())


def sign_receipt(receipt: dict[str, Any], signing_key) -> dict[str, Any]:
    receipt["prover"]["signature"] = "ed25519:" + signing_key.sign(receipt_hash(receipt)).signature.hex()
    return receipt


def check_receipt_signature(receipt: dict[str, Any]) -> tuple[bool, str]:
    from nacl.exceptions import BadSignatureError
    from nacl.signing import VerifyKey

    prover = receipt.get("prover") or {}
    pid, sig = prover.get("id", ""), prover.get("signature", "")
    if not (
        isinstance(pid, str) and pid.startswith("ed25519:") and isinstance(sig, str) and sig.startswith("ed25519:")
    ):
        return False, "missing ed25519 prover id or signature"
    try:
        VerifyKey(bytes.fromhex(pid[8:])).verify(receipt_hash(receipt), bytes.fromhex(sig[8:]))
    except (BadSignatureError, ValueError) as e:
        return False, f"signature does not verify: {e}"
    return True, "ok"


# ---------------------------------------------------------------------------
# Opening container: MAGIC || u64 len(index) || index JSON || safetensors blob
# ---------------------------------------------------------------------------


def tensor_key(pos: int, layer: int, name: str) -> str:
    return f"p{pos}/l{layer}/{name}"


def decode_step(n_prompt: int, n_gen: int, pos: int) -> int | None:
    """Index of the token sampled from the logits at forward position ``pos`` (None for prompt)."""
    t = pos - (n_prompt - 1)
    return t if 0 <= t < n_gen else None


def _int_list(xs: Any) -> list[int]:
    if not isinstance(xs, list) or not all(isinstance(x, int) and not isinstance(x, bool) for x in xs):
        raise ValueError(f"expected a list of integers, got {xs!r}")
    return sorted(set(xs))


def normalize_audits(challenge: dict[str, Any]) -> list[dict[str, Any]]:
    """The challenge's per-position audits, sorted by position, each ``{"pos", "layers", "attention"}``.

    Every audited position gets the embedding check, a full audit of its own layer subset (all
    tensors, bridges and both residual adds), the attention audit when ``attention``, and the
    decode checks when it samples a token.
    """
    audits = challenge["audits"]
    if not isinstance(audits, list) or not audits:
        raise ValueError("challenge needs a non-empty list of audits")
    out, seen = [], set()
    for a in audits:
        pos = a["pos"]
        if not isinstance(pos, int) or isinstance(pos, bool) or pos in seen:
            raise ValueError(f"bad or duplicate audit position {pos!r}")
        seen.add(pos)
        out.append({"pos": pos, "layers": _int_list(a["layers"]), "attention": bool(a.get("attention", False))})
    return sorted(out, key=lambda a: a["pos"])


def required_tensors(
    profile, n_prompt: int, n_gen: int, audits: list[dict[str, Any]]
) -> dict[int, dict[int, set[str]]]:
    """Which captured tensors an opening must contain, per position and layer group.

    For each audit: ``r_in`` of layer 0 (embedding check), the final group at decode positions
    (LM-head binding, final norm), every tensor of each audited layer, the next layer's ``r_in``
    (or ``r_final``) for the residual check, and with ``attention`` the ``k_n``/``v_n`` rows of
    every key position in that layer's attention window.
    """
    n_layers = profile.num_layers
    req: dict[int, dict[int, set[str]]] = {}

    def need(j: int, layer: int, names) -> None:
        req.setdefault(j, {}).setdefault(layer, set()).update(names)

    for a in audits:
        p = a["pos"]
        need(p, 0, ("r_in",))
        if decode_step(n_prompt, n_gen, p) is not None:
            need(p, n_layers, ("r_final", "h_final"))
        for layer in a["layers"]:
            need(p, layer, profile.capture_names(layer))
            if layer + 1 < n_layers:
                need(p, layer + 1, ("r_in",))
            else:
                need(p, n_layers, ("r_final",))
            if a["attention"]:
                for j in profile.window(layer, p):
                    need(j, layer, ("k_n", "v_n"))
    return req


def encode_opening(index: dict[str, Any], tensors: dict[str, np.ndarray]) -> bytes:
    from safetensors.numpy import save

    idx = json.dumps(index, sort_keys=True).encode()
    blob = save({k: np.ascontiguousarray(v) for k, v in tensors.items()}) if tensors else b""
    return OPENING_MAGIC + struct.pack("<Q", len(idx)) + idx + blob


def decode_opening(data: bytes) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    from safetensors.numpy import load

    if data[:8] != OPENING_MAGIC:
        raise ValueError("not a verifiable-gemma opening (bad magic)")
    (n,) = struct.unpack("<Q", data[8:16])
    index = json.loads(data[16 : 16 + n])
    blob = data[16 + n :]
    tensors = load(blob) if blob else {}
    return index, tensors
