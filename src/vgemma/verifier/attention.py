"""Attention audit: RoPE recomputation and single-query attention replay (audited, not verified).

Q and K are captured after q_norm / k_norm and before RoPE. The verifier applies RoPE from the
public parameters with the rounding of a bf16 eager forward (sliding layers: default RoPE over the
full head; global layers: proportional RoPE rotating the first 25 % of the pairs), computes scores
with scale 1.0 (QK-norm replaces 1/sqrt(d)), the causal and sliding-window mask, a float64 softmax
and the weighted sum of the V rows, and compares with the captured attention output ``a``.

Statistic: ``max_h ||a_h - ref_h||_2 / (||ref_h||_2 + ATTN_FLOOR sqrt(head_dim))``, bound ``ATTN_REL``.
"""

from __future__ import annotations

import numpy as np

from vgemma.canon import apply_rope_bf16, bf16_to_f64, rope_cos_sin
from vgemma.profile import GemmaProfile

# Set from measured honest deviations with margin (docs/DECISIONS.md); printed in every verdict.
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
        w = np.exp(s - s.max())
        w /= w.sum()
        out[h] = w @ v[:, g, :]
    return out.reshape(-1)


def deviation(profile: GemmaProfile, layer: int, a_bits: np.ndarray, ref: np.ndarray) -> float:
    hd = profile.head_dims[layer]
    a = bf16_to_f64(a_bits).reshape(-1, hd)
    r = ref.reshape(-1, hd)
    err = np.linalg.norm(a - r, axis=1)
    return float(np.max(err / (np.linalg.norm(r, axis=1) + ATTN_FLOOR * np.sqrt(hd))))
