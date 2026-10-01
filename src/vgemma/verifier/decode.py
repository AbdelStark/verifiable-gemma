"""Decode verification helpers: the manifest policy and sampled-token replay with the shared sampler."""

from __future__ import annotations

from typing import Any

import numpy as np

from vgemma.prover.sampler import SamplingPolicy, sample, uniform_for_step


def policy_from_manifest(manifest: dict[str, Any]) -> SamplingPolicy:
    return SamplingPolicy(
        temperature=float(manifest["temperature"]),
        top_k=int(manifest["top_k"]),
        top_p=float(manifest["top_p"]),
        greedy=bool(manifest["greedy"]),
    )


def replay_token(postcap: np.ndarray, policy: SamplingPolicy, seed: bytes, step: int) -> tuple[int, float]:
    u = uniform_for_step(seed, step)
    return sample(postcap, policy, u), u
