"""The sampler shared by prover and verifier (TECH_SPEC section 9).

Deterministic on every platform: float64 after the f32 input, `canon.exp_f64` instead of libm,
sequential cumulative sums, stable sorts with lowest-index tie-breaking, and a per-step uniform
draw derived from the revealed seed by SHA-256 (no dependence on a numpy RNG stream).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from vgemma.canon import H, exp_f64, tensor_hash, u32


@dataclass(frozen=True)
class SamplingPolicy:
    temperature: float = 1.0
    top_k: int = 64
    top_p: float = 0.95
    greedy: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def uniform_for_step(seed: bytes, step: int) -> float:
    """``u_t in [0, 1)`` with 53 random bits from ``H("vg/u" || seed || t)``."""
    d = H("vg/u", seed, u32(step))
    return (int.from_bytes(d[:8], "little") >> 11) / float(1 << 53)


def descending_order(z: np.ndarray, k: int) -> np.ndarray:
    """Exactly ``np.argsort(-z, kind="stable")[:k]`` (all when ``k <= 0``), in O(n) for small ``k``."""
    n = z.size
    if k <= 0 or k >= n:
        return np.argsort(-z, kind="stable")
    kth = np.partition(z, n - k)[n - k]  # the k-th largest value
    above = np.flatnonzero(z > kth)
    ties = np.flatnonzero(z == kth)[: k - above.size]  # lowest indices first, as a stable sort keeps them
    cand = np.concatenate([above, ties])
    return cand[np.lexsort((cand, -z[cand]))]


def sample(logits_postcap: np.ndarray, policy: SamplingPolicy, u: float) -> int:
    z = np.asarray(logits_postcap, dtype=np.float32).astype(np.float64)
    if policy.greedy or policy.temperature == 0.0:
        return int(np.argmax(z))  # first maximum: lowest index wins ties
    z = z / float(policy.temperature)
    order = descending_order(z, policy.top_k)  # descending, ties by lower index, first top_k
    zk = z[order]
    e = exp_f64(zk - zk[0])
    if policy.top_p < 1.0:
        cum = np.cumsum(e)
        n_keep = int(np.searchsorted(cum, float(policy.top_p) * cum[-1], side="left")) + 1
        order, e = order[:n_keep], e[:n_keep]
    by_index = np.argsort(order, kind="stable")  # inverse CDF over the kept set in index order
    kept, ek = order[by_index], e[by_index]
    cdf = np.cumsum(ek)
    j = int(np.searchsorted(cdf, u * cdf[-1], side="right"))
    return int(kept[min(j, len(kept) - 1)])


def sample_step(logits_postcap: np.ndarray, policy: SamplingPolicy, seed: bytes, step: int) -> tuple[int, dict]:
    """Sample one token; returns the token and the witness committed in the position leaf."""
    u = uniform_for_step(seed, step)
    token = sample(logits_postcap, policy, u)
    witness = {"step": step, "u": u, "postcap": tensor_hash(np.asarray(logits_postcap, dtype=np.float32)).hex()}
    return token, witness
