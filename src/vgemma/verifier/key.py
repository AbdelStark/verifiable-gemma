"""The secret verifier key (``key.npz``), loaded lazily so memory stays bounded."""

from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from vgemma.canon import f32_to_bf16
from vgemma.profile import GemmaProfile


class VerifierKey:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._npz = np.load(self.path, allow_pickle=False)
        self.meta: dict[str, Any] = json.loads(bytes(self._npz["meta"]).decode())
        self.profile = GemmaProfile.from_dict(self.meta["profile"])
        self.k = int(self.meta["freivalds_k"])

    def r(self, layer: int, family: str) -> np.ndarray:
        return self._npz[f"r.{layer}.{family}"]

    def v(self, layer: int, family: str) -> np.ndarray:
        return self._npz[f"v.{layer}.{family}"]

    def norm(self, layer: int, name: str) -> np.ndarray:
        return self._npz[f"w.{layer}.{name}"].astype(np.float64)

    def layer_scalar_bits(self, layer: int) -> np.ndarray:
        return f32_to_bf16(self._npz[f"s.{layer}"])

    @cached_property
    def final_norm(self) -> np.ndarray:
        return self._npz["w.final"].astype(np.float64)

    @cached_property
    def r_lm(self) -> np.ndarray:
        return self._npz["r.lm"]

    @cached_property
    def v_lm(self) -> np.ndarray:
        return self._npz["v.lm"]

    @property
    def size_bytes(self) -> int:
        return self.path.stat().st_size
