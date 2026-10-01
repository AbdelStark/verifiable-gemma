"""Canonical serialisation, hashing and reference functions shared by prover and verifier.

Two kinds of functions live here:

* Exact, platform-independent functions (`softcap`, `exp_f64`, bf16 rounding, hashing). They use
  only IEEE-754 correctly rounded operations (+, -, *, /, rint, ldexp) in a fixed order, so the
  prover and the verifier get bit-identical results on any machine.
* Float64 reference functions (`rmsnorm_gemma`, `gelu_tanh`, RoPE) that the verifier compares
  against captured bf16 tensors within a stated tolerance.

Convention: a numpy ``uint16`` array always holds bfloat16 bit patterns.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

DIGEST_SIZE = 32
EMPTY = bytes(DIGEST_SIZE)


def H(tag: str, *parts: bytes) -> bytes:
    """Domain-separated SHA-256: ``sha256(len(tag) || tag || parts...)``."""
    t = tag.encode("ascii")
    h = hashlib.sha256()
    h.update(bytes([len(t)]))
    h.update(t)
    for p in parts:
        h.update(p)
    return h.digest()


def u32(x: int) -> bytes:
    return struct.pack("<I", x)


def i32(x: int) -> bytes:
    return struct.pack("<i", x)


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def json_hash(tag: str, obj: Any) -> bytes:
    return H(tag, canonical_json(obj).encode("utf-8"))


# ---------------------------------------------------------------------------
# Canonical tensor bytes
# ---------------------------------------------------------------------------

_DTYPE_TAGS: dict[str, int] = {
    "bf16": 1,
    "f32": 2,
    "f64": 3,
    "i32": 4,
    "i64": 5,
    "u8": 6,
    "i8": 7,
    "f16": 8,
    "bool": 9,
}

_NP_TO_TAG: dict[np.dtype, str] = {
    np.dtype(np.uint16): "bf16",
    np.dtype(np.float32): "f32",
    np.dtype(np.float64): "f64",
    np.dtype(np.int32): "i32",
    np.dtype(np.int64): "i64",
    np.dtype(np.uint8): "u8",
    np.dtype(np.int8): "i8",
    np.dtype(np.float16): "f16",
    np.dtype(np.bool_): "bool",
}


def dtype_tag(arr: np.ndarray) -> str:
    try:
        return _NP_TO_TAG[arr.dtype.newbyteorder("=")]
    except KeyError as e:
        raise TypeError(f"no canonical dtype tag for {arr.dtype}") from e


def tensor_parts(arr: np.ndarray) -> tuple[bytes, memoryview]:
    """Header and zero-copy little-endian payload of the canonical tensor bytes."""
    tag = dtype_tag(arr)
    header = struct.pack("<BI", _DTYPE_TAGS[tag], arr.ndim) + struct.pack(f"<{arr.ndim}I", *arr.shape)
    le = np.ascontiguousarray(arr, dtype=arr.dtype.newbyteorder("<"))
    return header, memoryview(le.reshape(-1)).cast("B")


def tensor_bytes(arr: np.ndarray) -> bytes:
    """``dtype_tag (u8) || ndim (u32) || shape (u32 each) || raw little-endian bytes``."""
    header, payload = tensor_parts(arr)
    return header + payload.tobytes()


def tensor_hash(arr: np.ndarray) -> bytes:
    return H("vg/tensor", *tensor_parts(arr))


# ---------------------------------------------------------------------------
# bfloat16
# ---------------------------------------------------------------------------

BF16_U = 2.0**-8  # unit roundoff (8 significand bits, round to nearest)
F32_U = 2.0**-24
BF16_TINY = 2.0**-126


def bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint16)
    return (bits.astype(np.uint32) << 16).view(np.float32)


def bf16_to_f64(bits: np.ndarray) -> np.ndarray:
    return bf16_to_f32(bits).astype(np.float64)


def f32_to_bf16(x: np.ndarray) -> np.ndarray:
    """Round float32 to bfloat16 bits, round-to-nearest-even, NaN preserved (as torch does)."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    b = x.view(np.uint32)
    rounded = ((b + np.uint32(0x7FFF) + ((b >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)).astype(np.uint16)
    nan = np.isnan(x)
    if nan.any():
        rounded = np.where(nan, np.uint16(0x7FC0), rounded)
    return rounded


def bf16_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``bf16(a + b)`` with f32 opmath, the semantics of a torch bf16 add on CPU and GPU."""
    return f32_to_bf16(bf16_to_f32(a) + bf16_to_f32(b))


def bf16_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``bf16(a * b)`` with f32 opmath; the product of two bf16 values is exact in f32."""
    return f32_to_bf16(bf16_to_f32(a) * bf16_to_f32(b))


def bf16_ulp_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance in bf16 ulps between two bf16 arrays (monotone integer mapping of the bits)."""

    def ordinal(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.uint16).astype(np.int32)
        return np.where(x & 0x8000, 0x8000 - x, x)

    return np.abs(ordinal(a) - ordinal(b))


def embed_scale_bf16(hidden_size: int) -> float:
    """``sqrt(hidden)`` as torch computes it (f32 tensor) then cast to bf16."""
    return float(bf16_to_f32(f32_to_bf16(np.array([hidden_size**0.5], dtype=np.float32)))[0])


# ---------------------------------------------------------------------------
# Exact, deterministic transcendental functions (prover and verifier share these)
# ---------------------------------------------------------------------------

_LN2_HI = 6.93147180369123816490e-01  # fdlibm Cody-Waite split: k * _LN2_HI is exact for |k| < 2^20
_LN2_LO = 1.90821492927058770002e-10
_INV_LN2 = 1.44269504088896338700e00
_EXP_COEFFS = [1.0 / math.factorial(n) for n in range(13, -1, -1)]


def exp_f64(x: np.ndarray) -> np.ndarray:
    """exp(x) from correctly rounded basic ops only, so every platform agrees bit for bit.

    Range reduction ``x = k ln2 + r`` with ``|r| <= ln2/2``, degree-13 Taylor polynomial in Horner
    form (truncation error below 1e-17 relative), exact scaling by ``2^k``.
    """
    x = np.clip(np.asarray(x, dtype=np.float64), -1000.0, 709.0)
    k = np.rint(x * _INV_LN2)
    r = (x - k * _LN2_HI) - k * _LN2_LO
    p = np.full_like(r, _EXP_COEFFS[0])
    for c in _EXP_COEFFS[1:]:
        p = p * r + c
    return np.ldexp(p, k.astype(np.int64))


def tanh_f64(z: np.ndarray) -> np.ndarray:
    """Deterministic tanh built on `exp_f64`."""
    z = np.asarray(z, dtype=np.float64)
    a = np.abs(z)
    t = exp_f64(-2.0 * a)
    y = (1.0 - t) / (1.0 + t)
    a2 = a * a
    series = a * (1.0 + a2 * (-1.0 / 3.0 + a2 * (2.0 / 15.0)))
    y = np.where(a < 2.0**-10, series, y)
    return np.copysign(y, z)


def softcap(logits: np.ndarray, cap: float) -> np.ndarray:
    """Gemma final-logit soft-cap ``cap * tanh(logits / cap)``: f32 in, f32 out, bit-exact."""
    z = np.asarray(logits, dtype=np.float32).astype(np.float64) / float(cap)
    return (float(cap) * tanh_f64(z)).astype(np.float32)


# ---------------------------------------------------------------------------
# Float64 reference functions for tolerance-bounded replay
# ---------------------------------------------------------------------------


def rmsnorm_gemma(x: np.ndarray, weight: np.ndarray | None, eps: float) -> np.ndarray:
    """Gemma 4 RMSNorm over the last axis: ``x * (mean(x^2) + eps)^-0.5 * w`` (w optional).

    Note: Gemma 4 multiplies by ``w`` directly; the Gemma 1-3 ``(1 + w)`` form does not apply.
    """
    x = np.asarray(x, dtype=np.float64)
    ms = np.mean(x * x, axis=-1, keepdims=True) + eps
    y = x / np.sqrt(ms)
    if weight is not None:
        y = y * np.asarray(weight, dtype=np.float64)
    return y


_GELU_C = math.sqrt(2.0 / math.pi)


def gelu_tanh(x: np.ndarray) -> np.ndarray:
    """``gelu_pytorch_tanh``: ``0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))`` in float64."""
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * x * (1.0 + np.tanh(_GELU_C * (x + 0.044715 * x * x * x)))


# ---------------------------------------------------------------------------
# RoPE (Gemma 4: default RoPE on sliding layers, proportional partial RoPE on global layers)
# ---------------------------------------------------------------------------


def rope_inv_freq(rope_params: dict[str, Any], head_dim: int) -> np.ndarray:
    """Inverse frequencies as transformers computes them (float32), length ``head_dim // 2``.

    * ``default``: ``1 / theta^(2i / head_dim)`` for every pair.
    * ``proportional``: the first ``int(partial_rotary_factor * head_dim // 2)`` pairs rotate with
      exponents still divided by the full ``head_dim``; the remaining pairs get frequency 0 (cos 1,
      sin 0), so they pass through unrotated.
    """
    rope_type = rope_params.get("rope_type", "default")
    theta = np.float32(rope_params["rope_theta"])
    dim = np.float32(head_dim)
    if rope_type == "default":
        exps = np.arange(0, head_dim, 2, dtype=np.float32) / dim
        return (np.float32(1.0) / np.power(theta, exps)).astype(np.float32)
    if rope_type == "proportional":
        factor = np.float32(rope_params.get("factor", 1.0))
        proportion = float(rope_params.get("partial_rotary_factor", 1.0))
        rope_angles = int(proportion * head_dim // 2)
        exps = np.arange(0, 2 * rope_angles, 2, dtype=np.int64).astype(np.float32) / dim
        rotated = (np.float32(1.0) / np.power(theta, exps)).astype(np.float32)
        nope = head_dim // 2 - rope_angles
        inv = np.concatenate([rotated, np.zeros(nope, dtype=np.float32)]) if nope > 0 else rotated
        return (inv / factor).astype(np.float32)
    raise ValueError(f"unsupported rope_type {rope_type!r}")


def rope_cos_sin(rope_params: dict[str, Any], head_dim: int, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """cos and sin tables ``[n_pos, head_dim]`` in float64 for the given absolute positions.

    The angle is formed in float32 (``inv_freq * pos``) like transformers; cos and sin are then
    evaluated in float64.
    """
    inv = rope_inv_freq(rope_params, head_dim)
    pos = np.asarray(positions, dtype=np.float32).reshape(-1, 1)
    freqs = (pos * inv[None, :]).astype(np.float32).astype(np.float64)
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb), np.sin(emb)


def rotate_half(x: np.ndarray) -> np.ndarray:
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def apply_rope_bf16(x_bits: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """RoPE with the rounding of a bf16 eager forward: cos/sin cast to bf16, then
    ``bf16(bf16(x * cos) + bf16(rotate_half(x) * sin))``. ``x_bits``: ``[n, heads, head_dim]``
    bf16 bits; ``cos``/``sin``: ``[n, head_dim]`` float64. Returns float64 values.
    """
    cos_b = f32_to_bf16(cos.astype(np.float32))[:, None, :]
    sin_b = f32_to_bf16(sin.astype(np.float32))[:, None, :]
    x = np.asarray(x_bits, dtype=np.uint16)
    rot = f32_to_bf16(rotate_half(bf16_to_f32(x)))
    t1 = bf16_mul(x, cos_b)
    t2 = bf16_mul(rot, sin_b)
    return bf16_to_f64(bf16_add(t1, t2))
