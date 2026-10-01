"""Freivalds checks with a bf16 output tolerance (TECH_SPEC section 11).

For ``y = W x`` computed with bf16 inputs, f32 accumulation and bf16 output rounding, and ``k``
secret Rademacher vectors ``r_j`` with ``v_j = r_j^T W`` precomputed in float64:

    d_j = r_j . y - v_j . x = r_j . e,     e = y - W x  (rounding and accumulation error)

The prover never sees ``r``, so ``e`` is independent of it: each ``d_j`` has mean 0 and variance
``||e||_2^2 <= (u ||y||_2)^2`` with ``u = 2^-8``. The statistic is ``T = sqrt(mean_j d_j^2)`` and
the bound is

    tau = (3 u + 8 sqrt(K) 2^-24) ||y||_2

(three times the worst-case rounding norm, plus a generous f32 accumulation allowance for an inner
dimension ``K``). For an honest prover ``T > tau`` requires a chi-square(k) variable to exceed
``9k``: with k = 16, probability below 1e-20 even when every element sits at half an ulp. A weight
change that moves ``y`` by ``delta`` gives ``T ~ ||delta||_2``, so changes above about 1.2 % of
``||y||_2`` are rejected with overwhelming probability.
"""

from __future__ import annotations

import math

import numpy as np

from vgemma.canon import BF16_U, F32_U

ROUNDING_MARGIN = 3.0
ACC_MARGIN = 8.0


def tolerance(y: np.ndarray, u_out: float, k_in: int) -> float:
    return (ROUNDING_MARGIN * u_out + ACC_MARGIN * math.sqrt(k_in) * F32_U) * float(np.linalg.norm(y))


def deviation(r: np.ndarray, v: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    d = r.astype(np.float64) @ y - v @ x
    return float(np.sqrt(np.mean(d * d)))


def check(r: np.ndarray, v: np.ndarray, x: np.ndarray, y: np.ndarray, u_out: float = BF16_U) -> tuple[float, float]:
    """Returns (deviation T, tolerance tau) for float64 ``x`` ``[in]`` and ``y`` ``[out]``."""
    if r.shape != (v.shape[0], y.shape[0]) or v.shape[1] != x.shape[0]:
        raise ValueError(f"Freivalds shape mismatch: r {r.shape}, v {v.shape}, x {x.shape}, y {y.shape}")
    return deviation(r, v, x, y), tolerance(y, u_out, x.shape[0])
