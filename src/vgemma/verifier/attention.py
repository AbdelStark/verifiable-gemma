"""Attention audit: RoPE recomputation and single-query attention replay (audited, not verified).

Q and K are captured after q_norm / k_norm and before RoPE. The verifier applies RoPE from the
public parameters with the rounding of a bf16 eager forward (sliding layers: default RoPE over the
full head; global layers: proportional RoPE rotating the first 25 % of the pairs), computes scores
with scale 1.0 (QK-norm replaces 1/sqrt(d)), the causal and sliding-window mask, the softmax
and the weighted sum of the V rows, and compares with the captured attention output ``a``.

The softmax and weighted sum follow the rounding of the declared ``attn_implementation``, measured
on CPU (tiny) and GPU (Gemma 4 12B, A100), docs/DECISIONS.md 44:

* ``sdpa``: f32 scores, the unnormalised probabilities ``P`` rounded to bf16 for the ``P V``
  product, normalised by the f32 sum (flash-attention style);
* ``eager``: scores rounded to bf16 (bf16 matmul output), softmax in f32, weights rounded to bf16.

Statistic: ``max_h ||a_h - ref_h||_2 / (||ref_h||_2 + ATTN_FLOOR sqrt(head_dim))``, bound ``ATTN_REL``.
"""

from __future__ import annotations

import numpy as np

from vgemma.canon import apply_rope_bf16, bf16_to_f64, f32_to_bf16, rope_cos_sin
from vgemma.profile import GemmaProfile

# Honest worst with the matching numerics: 0.0029 on tiny CPU, 0.0075 on Gemma 4 12B / A100 over 1,500+
# replays (p99 0.0036); the bound is about 4x the worst measured value (docs/DECISIONS.md 44).
ATTN_REL = 2.0**-5
ATTN_FLOOR = 2.0**-14


def replay(
    profile: GemmaProfile,
    layer: int,
    pos: int,
    q_n: np.ndarray,
    k_rows: np.ndarray,
    v_rows: np.ndarray,
    key_positions: list[int],
    attn_implementation: str = "sdpa",
) -> np.ndarray:
    """Recompute ``a`` ``[heads * head_dim]`` for query ``pos`` from pre-RoPE q_n ``[H, hd]`` and the
    window's k_n / v_n rows ``[n, KV, hd]`` (bf16 bits)."""
    hd, kv, n_heads = profile.head_dims[layer], profile.kv_heads[layer], profile.num_heads
    params = profile.rope_parameters[profile.layer_types[layer]]
    cos, sin = rope_cos_sin(params, hd, np.array([pos, *key_positions]))
    q = apply_rope_bf16(q_n[None], cos[:1], sin[:1])[0]  # [H, hd]
    k = apply_rope_bf16(k_rows, cos[1:], sin[1:])  # [n, KV, hd]
    v = bf16_to_f64(v_rows)
    group = n_heads // kv
    out = np.empty((n_heads, hd))
    for h in range(n_heads):
        g = h // group
        s = k[:, g, :] @ q[h]  # scaling 1.0
        if attn_implementation == "eager":
            s = _bf16(s)
            p = np.exp(s - s.max())
            out[h] = _bf16(p / p.sum()) @ v[:, g, :]
        else:
            p = np.exp(s - s.max())
            out[h] = (_bf16(p) @ v[:, g, :]) / p.sum()
    return out.reshape(-1)


def _bf16(x: np.ndarray) -> np.ndarray:
    return bf16_to_f64(f32_to_bf16(np.asarray(x, dtype=np.float32)))


def deviation(profile: GemmaProfile, layer: int, a_bits: np.ndarray, ref: np.ndarray) -> float:
    hd = profile.head_dims[layer]
    a = bf16_to_f64(a_bits).reshape(-1, hd)
    r = ref.reshape(-1, hd)
    err = np.linalg.norm(a - r, axis=1)
    return float(np.max(err / (np.linalg.norm(r, axis=1) + ATTN_FLOOR * np.sqrt(hd))))
