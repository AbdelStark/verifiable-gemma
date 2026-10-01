"""Real-model smoke test: the full demo on one GPU (skipped without CUDA).

VG_GPU_MODEL=google/gemma-4-12B-it uv run pytest -m gpu tests/test_gpu_smoke.py
"""

from __future__ import annotations

import os

import pytest


@pytest.mark.gpu
def test_demo_on_real_gemma4(tmp_path):
    from vgemma.demo import run_demo

    model = os.environ.get("VG_GPU_MODEL", "google/gemma-4-12B-it")
    summary = run_demo(model, tiny=False, device="cuda", workdir=tmp_path, max_new_tokens=16)
    assert summary["ok"], summary["results"]
    assert {r["scenario"] for r in summary["results"]} == {"honest", "weights", "identity", "sampling", "softcap"}
