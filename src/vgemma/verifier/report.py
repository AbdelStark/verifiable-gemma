"""Check ledger: records every check with its kind, observed deviation and bound; fails fast."""

from __future__ import annotations

import math
from typing import Any

from vgemma.verifier.codes import CODES, kind_of


class VerifyFail(Exception):
    def __init__(self, code: str, message: str, layer: int | None = None, position: int | None = None, **detail):
        if code not in CODES:
            raise ValueError(f"unknown reason code {code}")
        super().__init__(f"{code}: {message}")
        self.code, self.message, self.layer, self.position, self.detail = code, message, layer, position, detail


class Ledger:
    """Per reason code: kind (exact | tolerance | audited), count, worst deviation/bound ratio."""

    def __init__(self) -> None:
        self.stats: dict[str, dict[str, Any]] = {}
        self.ratios: dict[str, list[float]] = {}

    def _rec(self, code: str) -> dict[str, Any]:
        return self.stats.setdefault(code, {"kind": kind_of(code), "n": 0, "max_ratio": 0.0, "worst": None})

    def exact(self, code: str, ok: bool, message: str = "", layer=None, position=None, **detail) -> None:
        """A pass/fail check (bit-exact comparison, binding, presence)."""
        s = self._rec(code)
        s["n"] += 1
        if not ok:
            raise VerifyFail(code, message or CODES[code], layer, position, **detail)

    def tolerance(self, code: str, deviation: float, bound: float, layer=None, position=None, unit: str = "") -> None:
        """Pass iff both values are finite and ``deviation <= bound``. Both are kept for the report."""
        s = self._rec(code)
        s["n"] += 1
        deviation, bound = float(deviation), float(bound)
        finite = math.isfinite(deviation) and math.isfinite(bound)
        ratio = deviation / bound if finite and bound > 0 else (0.0 if finite and deviation == 0 else math.inf)
        self.ratios.setdefault(code, []).append(ratio)
        if s["worst"] is None or ratio >= s["max_ratio"]:
            s["max_ratio"] = ratio
            s["worst"] = {
                "deviation": deviation,
                "tolerance": bound,
                "layer": layer,
                "position": position,
                "unit": unit,
            }
        if not (finite and deviation <= bound):
            raise VerifyFail(
                code,
                f"deviation {deviation:.4g} exceeds tolerance {bound:.4g}{' ' + unit if unit else ''}",
                layer,
                position,
                deviation=deviation,
                tolerance=bound,
            )

    def summary(self) -> dict[str, Any]:
        """Per code: kind, count, worst; for tolerance checks also the deviation/bound distribution."""
        out = {k: dict(v) for k, v in self.stats.items()}
        for code, rs in self.ratios.items():
            xs = sorted(rs)

            def q(f: float, xs=xs) -> float:
                return xs[min(len(xs) - 1, int(f * len(xs)))]

            out[code]["ratio_quantiles"] = {"p50": q(0.5), "p90": q(0.9), "p99": q(0.99), "max": xs[-1]}
        return out
