"""Modal app for real-model runs (AGENTS.md, TECH_SPEC section 13b).

Volumes:
  vg-hf-cache  /cache   model weights (HF_HOME=/cache/hf), downloaded once
  vg-state     /state   public.json, retained state, receipts, openings, demo and bench outputs
  vg-keys      /keys    secret verifier keys; mounted only by keygen, demo and bench, never by serve

Commands (from the repository root):
  uv run modal run infra/modal_app.py::download --model google/gemma-4-12B-it
  uv run modal run --detach infra/modal_app.py::keygen --model google/gemma-4-12B-it
  uv run modal run infra/modal_app.py::demo --model google/gemma-4-12B-it
  uv run modal run infra/modal_app.py::bench --model google/gemma-4-12B-it
  uv run modal run infra/modal_app.py::tests_tiny
  VG_MODEL=google/gemma-4-12B-it uv run modal deploy infra/modal_app.py      # web server on port 8000
  VG_GPU=H100 uv run modal run infra/modal_app.py::demo --model google/gemma-4-31B-it
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "google/gemma-4-12B-it"
GPU = os.environ.get("VG_GPU", "A100-40GB")  # A100-40GB or L40S for 12B; A100-80GB or H100 for 31B
SERVE_MODEL = os.environ.get("VG_MODEL", DEFAULT_MODEL)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(uv_project_dir=str(REPO), extras=["gpu"], groups=["dev"])
    .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1", "VG_MODEL": SERVE_MODEL})
    .add_local_dir(str(REPO / "tests"), "/root/tests", copy=False)
    .add_local_file(str(REPO / "pyproject.toml"), "/root/pyproject.toml", copy=False)
    .add_local_python_source("vgemma")
)

app = modal.App("verifiable-gemma", image=image)
hf_cache = modal.Volume.from_name("vg-hf-cache", create_if_missing=True)
state = modal.Volume.from_name("vg-state", create_if_missing=True)
keys = modal.Volume.from_name("vg-keys", create_if_missing=True)


def _name(model: str) -> str:
    return model.replace("/", "--")


def _public_path(model: str) -> Path:
    return Path("/state") / _name(model) / "public.json"


def _key_dir(model: str) -> Path:
    return Path("/keys") / _name(model)


@app.function(volumes={"/cache": hf_cache}, timeout=4 * 3600, cpu=4.0, memory=8192)
def download(model: str = DEFAULT_MODEL) -> str:
    from vgemma.model import resolve_model_dir

    path = str(resolve_model_dir(model))
    hf_cache.commit()
    print(f"{model} cached at {path}")
    return path


@app.function(cpu=4.0, memory=16384, timeout=1800)
def tests_tiny() -> None:
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "/root/tests"]
    subprocess.run(cmd, check=True, cwd="/root")


@app.function(volumes={"/cache": hf_cache, "/state": state, "/keys": keys}, cpu=8.0, memory=65536, timeout=6 * 3600)
def keygen(model: str = DEFAULT_MODEL, k: int = 16) -> dict:
    from vgemma.keygen import keygen as run

    public = run(model, _key_dir(model), k=k, public_out=_public_path(model))
    keys.commit()
    state.commit()
    return public


@app.function(
    gpu=GPU,
    volumes={"/cache": hf_cache, "/state": state, "/keys": keys},
    cpu=8.0,
    memory=65536,
    timeout=3 * 3600,
    scaledown_window=60,
)
def demo(model: str = DEFAULT_MODEL, max_new_tokens: int = 24, tamper_layers: str = "full") -> dict:
    """The provider runs as a subprocess that is given only /state paths; this process holds the key."""
    from vgemma.demo import run_demo

    workdir = Path("/state/demo") / _name(model)
    summary = run_demo(
        model,
        tiny=False,
        device="cuda",
        workdir=workdir,
        max_new_tokens=max_new_tokens,
        tamper_layers=tamper_layers,
        key_dir=_key_dir(model),
        public_path=_public_path(model),
    )
    state.commit()
    keys.commit()
    if not summary["ok"]:
        raise RuntimeError("demo outcomes did not match expectations; see /state/demo logs")
    return summary


@app.function(
    gpu=GPU,
    volumes={"/cache": hf_cache, "/state": state, "/keys": keys},
    cpu=8.0,
    memory=65536,
    timeout=3 * 3600,
    scaledown_window=60,
)
def bench(model: str = DEFAULT_MODEL, max_new_tokens: int = 32, runs: int = 3, calibrate: bool = True) -> dict:
    from vgemma.demo import run_bench

    out = run_bench(
        model,
        tiny=False,
        device="cuda",
        workdir=Path("/state/bench") / _name(model),
        max_new_tokens=max_new_tokens,
        runs=runs,
        calibrate=calibrate,
        key_dir=_key_dir(model),
        public_path=_public_path(model),
    )
    state.commit()
    keys.commit()
    return out


@app.function(
    gpu=GPU,
    volumes={"/cache": hf_cache, "/state": state},  # never /keys
    cpu=8.0,
    memory=65536,
    timeout=24 * 3600,
    scaledown_window=120,
)
@modal.web_server(8000, startup_timeout=1800)
def serve() -> None:
    model = os.environ["VG_MODEL"]
    subprocess.Popen(
        [
            sys.executable, "-m", "vgemma.cli", "serve", "--model", model,
            "--public", str(_public_path(model)), "--state-dir", "/state/serve/" + _name(model),
            "--host", "0.0.0.0", "--port", "8000", "--device", "cuda",
        ]
    )  # fmt: skip
