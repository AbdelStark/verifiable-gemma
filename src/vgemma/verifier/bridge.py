"""Bridge replay: norms, GELU gate (tolerance-bounded, elementwise) and the residual chain (exact).

Elementwise bounds, with u = 2^-8 the bf16 unit roundoff:

* RMSNorm (f32 compute, one bf16 rounding): ``|y - ref| <= 2u |ref| + 2^-126``.
* GELU gate ``h = bf16(bf16(gelu(g)) * u)`` (two roundings): ``|h - ref| <= 4u |ref| + 2^-20 |g u|
  + 2^-126``; the middle term covers f32 cancellation in ``1 + tanh`` for very negative ``g``.

Each check reports ``max_i |y_i - ref_i| / bound_i`` against 1.0.
"""

from __future__ import annotations

import numpy as np

from vgemma.canon import BF16_TINY, BF16_U, bf16_add, bf16_mul, bf16_to_f64, gelu_tanh, rmsnorm_gemma

NORM_REL = 2 * BF16_U
GELU_REL = 4 * BF16_U
GELU_CANCEL = 2.0**-20


def bounded_ratio(obs: np.ndarray, ref: np.ndarray, bound: np.ndarray) -> float:
    r = np.abs(obs - ref) / bound
    return float(np.max(r)) if r.size else 0.0


def norm_ratio(x_bits: np.ndarray, y_bits: np.ndarray, weight: np.ndarray | None, eps: float) -> float:
    """RMSNorm over the last axis of ``x``; ``weight`` may be None (v_norm)."""
    ref = rmsnorm_gemma(bf16_to_f64(x_bits), weight, eps)
    return bounded_ratio(bf16_to_f64(y_bits), ref, NORM_REL * np.abs(ref) + BF16_TINY)


def gelu_ratio(g_bits: np.ndarray, u_bits: np.ndarray, h_bits: np.ndarray) -> float:
    g, u = bf16_to_f64(g_bits), bf16_to_f64(u_bits)
    ref = gelu_tanh(g) * u
    bound = GELU_REL * np.abs(ref) + GELU_CANCEL * np.abs(g * u) + BF16_TINY
    return bounded_ratio(bf16_to_f64(h_bits), ref, bound)


def residual_mismatches(a_bits: np.ndarray, b_bits: np.ndarray, out_bits: np.ndarray, scalar_bits=None) -> int:
    """Exact: ``out == bf16(a + b)`` (times ``layer_scalar`` with one more rounding when given)."""
    ref = bf16_add(a_bits, b_bits)
    if scalar_bits is not None:
        ref = bf16_mul(ref, scalar_bits)
    return int(np.count_nonzero(ref != np.asarray(out_bits, dtype=np.uint16)))
