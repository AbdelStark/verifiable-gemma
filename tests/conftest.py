"""Shared tiny-mode fixtures: one tiny checkpoint, one key, engines per tamper configuration."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from vgemma.keygen import keygen
from vgemma.model import load_model
from vgemma.prover.engine import Engine, ProverIdentity
from vgemma.prover.store import RetainedStore
from vgemma.prover.tamper import Tamper
from vgemma.tiny import build_tiny
from vgemma.tokenizer import load_tokenizer
from vgemma.verifier.key import VerifierKey


def pytest_collection_modifyitems(config, items):
    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        has_cuda = False
    skip = pytest.mark.skip(reason="needs a CUDA device")
    for item in items:
        if "gpu" in item.keywords and not has_cuda:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def tiny_dir(tmp_path_factory) -> Path:
    return build_tiny(tmp_path_factory.mktemp("tiny") / "tiny-gemma4")


@pytest.fixture(scope="session")
def keys(tiny_dir, tmp_path_factory) -> tuple[VerifierKey, dict[str, Any]]:
    out = tmp_path_factory.mktemp("keys")
    public = keygen(str(tiny_dir), out, log=lambda *_: None)
    return VerifierKey(out / "key.npz"), public


@pytest.fixture(scope="session")
def state_root(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("state")


@pytest.fixture(scope="session")
def make_engine(tiny_dir, keys, state_root) -> Callable[..., Engine]:
    _, public = keys
    identity = ProverIdentity.load_or_create(state_root / "prover_identity.json")

    def factory(tamper: str | Tamper | None = None, patch: Callable | None = None, attn: str = "sdpa") -> Engine:
        lm = load_model(str(tiny_dir), attn_implementation=attn)
        if patch is not None:
            patch(lm)
        tok = load_tokenizer(lm.model_dir, lm.eos_token_ids)
        t = Tamper(tamper) if isinstance(tamper, str) else tamper
        return Engine(lm, tok, RetainedStore(state_root / "retained"), identity, public, tamper=t, log=lambda *_: None)

    return factory


@pytest.fixture(scope="session")
def engine(make_engine) -> Engine:
    return make_engine()
