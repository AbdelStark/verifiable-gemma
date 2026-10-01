"""Honest runs in tiny mode: full and routine layers, greedy and sampled, stops, multi-position, HTTP."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from helpers import PROMPT, Run, check, dump
from vgemma.auditor import make_challenge
from vgemma.protocol import decode_opening
from vgemma.prover.sampler import SamplingPolicy
from vgemma.prover.server import create_app


@pytest.mark.parametrize("layers", ["full", "routine", "routine:1", "5", "0,3"])
def test_honest_layers(engine, keys, layers):
    run = Run(engine, layers=layers)
    v = check(keys, run.receipt, run.opening, run.challenge)
    assert v["result"] == "PASS", dump(v)
    assert v["coverage"]["attention"].startswith("audited")


@pytest.mark.parametrize(
    "policy",
    [
        SamplingPolicy(),
        SamplingPolicy(greedy=True),
        SamplingPolicy(temperature=0.7, top_k=0, top_p=1.0),
        SamplingPolicy(temperature=1.3, top_k=5, top_p=0.5),
    ],
)
def test_honest_sampling_policies(engine, keys, policy):
    run = Run(engine, policy=policy, positions="all-gen", layers="5")
    v = check(keys, run.receipt, run.opening, run.challenge)
    assert v["result"] == "PASS", dump(v)
    assert v["checks"]["DECODE_SAMPLING"]["n"] == 2 * run.receipt["n_gen"]


def test_max_token_stop(engine, keys):
    run = Run(engine, max_new_tokens=5)
    assert run.receipt["n_gen"] == 5 or run.trace.tokens[-1] in engine.lm.eos_token_ids
    assert check(keys, run.receipt, run.opening, run.challenge)["result"] == "PASS"


def test_eos_stop(engine, keys):
    eos = set(engine.lm.eos_token_ids)
    for i in range(200):
        run = Run(engine, max_new_tokens=48, request_id=f"r_eos{i}", positions="random:2", layers="routine:2")
        if run.trace.tokens[-1] in eos and run.receipt["n_gen"] < 48:
            break
    else:
        pytest.skip("no EOS sampled in 200 tries")
    v = check(keys, run.receipt, run.opening, run.challenge)
    assert v["result"] == "PASS", dump(v)


def test_single_token_generation(engine, keys):
    run = Run(engine, max_new_tokens=1)
    assert run.receipt["n_gen"] == 1
    assert check(keys, run.receipt, run.opening, run.challenge)["result"] == "PASS"


def test_multi_position_including_prefill(engine, keys):
    run = Run(engine, positions="0,1,7,8,9,20", layers="full", decode="none")
    v = check(keys, run.receipt, run.opening, run.challenge)
    assert v["result"] == "PASS", dump(v)
    assert "DECODE_SAMPLING" not in v["checks"]  # prompt positions sample nothing
    run2 = Run(engine, positions="all-gen", layers="routine:2")
    assert check(keys, run2.receipt, run2.opening, run2.challenge)["result"] == "PASS"


def test_window_boundaries(engine, keys):
    """Sliding window 8: query positions on both sides of the window edge, global layer full prefix."""
    run = Run(engine, positions="6,7,8,15,16,30", layers="0,5")
    v = check(keys, run.receipt, run.opening, run.challenge)
    assert v["result"] == "PASS", dump(v)


def test_without_attention_audit(engine, keys):
    run = Run(engine, attention=False)
    v = check(keys, run.receipt, run.opening, run.challenge)
    assert v["result"] == "PASS" and "ATTN_REPLAY" not in v["checks"]
    assert len(run.opening) < len(Run(engine).opening)


def test_eager_attention_path(make_engine, keys):
    eng = make_engine(attn="eager")
    run = Run(eng)
    assert run.receipt["manifest"]["attn_implementation"] == "eager"
    assert check(keys, run.receipt, run.opening, run.challenge)["result"] == "PASS"


def test_same_request_reproduces_tokens(engine):
    a = Run(engine, request_id="r_repro01")
    b = Run(engine, request_id="r_repro01")
    assert a.trace.tokens == b.trace.tokens
    assert a.receipt["trace_root"] == b.receipt["trace_root"]
    assert a.receipt["io_chain_head"] == b.receipt["io_chain_head"]
    c = Run(engine, request_id="r_repro02")
    assert c.receipt["seed_commitment"] != a.receipt["seed_commitment"]


def test_retained_state_reload(engine, make_engine, keys):
    run = Run(engine)
    other = make_engine()  # a fresh engine on the same store opens an earlier request
    opening = other.open(run.receipt["request_id"], run.challenge)
    assert check(keys, run.receipt, opening, run.challenge)["result"] == "PASS"


def test_opening_contents_are_minimal(engine):
    run = Run(engine, positions="gen:0", layers="2", attention=False, decode="none")
    index, tensors = decode_opening(run.opening)
    p = run.full_positions[0]
    assert sorted(int(k) for k, e in index["positions"].items() if "layers" in e) == [p]
    token_only = [e for e in index["positions"].values() if "layers" not in e]
    assert len(token_only) == len(index["positions"]) - 1 and all(
        set(e) == {"input_token", "body", "proof"} for e in token_only
    )
    layer_keys = {k.split("/")[1] for k in tensors if k.startswith(f"p{p}/") and k.count("/") == 2}
    assert layer_keys == {"l0", "l2", "l3", "l6"}  # embedding r_in, challenged layer, next r_in, final group
    assert {f"p{p}/logits_precap", f"p{p}/embed_row"} <= set(tensors)
    assert not any(k.endswith("logits_postcap") for k in tensors)  # recomputed by the verifier


def test_http_round_trip(engine, keys):
    client = TestClient(create_app(engine))
    h = client.get("/health").json()
    assert h["model"]["weights_root"] == keys[1]["weights_root"]
    resp = client.post("/chat", json={"prompt": PROMPT, "max_new_tokens": 8}).json()
    receipt = resp["receipt"]
    ch = make_challenge(receipt, engine.profile.num_layers, layers="routine")
    opening = client.post("/audit", json=ch).content
    v = check(keys, receipt, opening, ch, prover_id=h["prover_id"])
    assert v["result"] == "PASS", dump(v)
    assert client.post("/audit", json={**ch, "request_id": "r_unknown"}).status_code == 404
    bad = client.post("/audit", json={**ch, "audits": [{"pos": 10_000, "layers": [0], "attention": False}]})
    assert bad.status_code == 400


def test_verifier_needs_no_torch():
    import subprocess
    import sys

    code = "import sys, vgemma.verifier.verify, vgemma.verifier.key; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "False"
