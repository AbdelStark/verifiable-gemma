"""Regressions for the soundness review (each test is a former PoC that made the verifier PASS)."""

from __future__ import annotations

import copy
import secrets

import numpy as np
import pytest

from helpers import PROMPT, Run, check, dump, mutate_opening, resign
from vgemma.auditor import choose_positions, decode_positions, make_challenge
from vgemma.canon import softcap, tensor_hash
from vgemma.protocol import decode_opening, encode_opening, tensor_key
from vgemma.prover.engine import build_opening
from vgemma.prover.sampler import SamplingPolicy
from vgemma.verifier.report import Ledger, VerifyFail


def expect(v, code):
    assert v["result"] == "FAIL", f"expected {code}, got PASS"
    assert v["reason"] == code, f"expected {code}, got {dump(v)}"


def wrap_logits(engine, fn):
    """A malicious prover that edits its own logits before soft-cap and sampling."""
    orig = engine._logits
    state = {"t": 0}

    def logits(h):
        z = orig(h)
        out = fn(z.copy(), state["t"])
        state["t"] += 1
        return out

    engine._logits = logits
    return engine


# 1. non-finite values -----------------------------------------------------------------------------


def test_tolerance_rejects_non_finite():
    for dev, bound in ((np.inf, np.inf), (np.nan, 1.0), (0.0, np.inf), (np.inf, 1.0)):
        with pytest.raises(VerifyFail):
            Ledger().tolerance("LMHEAD_BINDING", dev, bound)
    Ledger().tolerance("LMHEAD_BINDING", 0.0, 1.0)


def test_infinite_logit_forcing_a_token(make_engine, keys):
    target = [ord(c) + 16 for c in "PWNED!"]

    def force(z, t):
        z[target[t % len(target)]] = np.inf
        return z

    eng = wrap_logits(make_engine(), force)
    r = Run(eng, positions="random", layers="full")
    expect(check(keys, r.receipt, r.opening, r.challenge, prover_id=eng.identity.id), "OPENING_SCHEMA")


# 2. a layer group carrying both the committed leaf and fabricated tensors -------------------------


def test_leaf_plus_fabricated_tensors(make_engine, keys):
    eng = make_engine()
    r = Run(eng)
    trace, com = eng.store.load(r.receipt["request_id"])
    p = r.full_positions[0]

    def both(idx, t):
        g = idx["positions"][str(p)]["layers"]["2"]
        g["leaf"] = bytes(com.layer_leaves[p, 2]).hex()
        t[tensor_key(p, 2, "a")][0] ^= 1

    expect(check(keys, r.receipt, mutate_opening(r.opening, both), r.challenge), "OPENING_SCHEMA")


# 3. predictable challenge positions --------------------------------------------------------------


def test_default_positions_are_random(engine):
    r = Run(engine, max_new_tokens=16).receipt
    draws = {tuple(a["pos"] for a in make_challenge(r, 6)["audits"] if a["attention"]) for _ in range(30)}
    assert len(draws) > 5 and all(set(d) <= set(decode_positions(r)) for d in draws)
    assert choose_positions("edges", r) == choose_positions("edges", r)


def test_tokens_forged_outside_challenged_positions(make_engine, keys):
    """Forging every token except first, middle and last: the decode checks cover all tokens."""
    n = 12
    safe = {0, n // 2, n - 1}

    def forge(z, t):
        if t not in safe:
            z[ord("x") + 16] += 1e4
        return z

    eng = wrap_logits(make_engine(), forge)
    r = Run(eng, max_new_tokens=n, positions="edges", layers="full")
    v = check(keys, r.receipt, r.opening, r.challenge)
    expect(v, "LMHEAD_BINDING")
    assert v["position"] not in r.full_positions  # caught at a decode-only position


# 4. prompt and request binding -------------------------------------------------------------------


def test_prompt_must_be_opened(engine, keys):
    r = Run(engine)
    op = mutate_opening(r.opening, lambda idx, t: idx.__setitem__("prompt_tokens", None))
    expect(check(keys, r.receipt, op, r.challenge), "PROMPT_BINDING")


def test_prompt_substitution_against_client_request(engine, keys):
    real = engine.tokenizer.apply_chat([{"role": "user", "content": PROMPT}])
    fake = list(real)
    fake[5:12] = [ord(c) + 16 for c in "IGNORE!"]
    res = engine.generate(prompt_tokens=fake, max_new_tokens=8)
    ch = make_challenge(res.receipt, 6, layers="full")
    op = engine.open(res.receipt["request_id"], ch)
    assert check(keys, res.receipt, op, ch)["result"] == "PASS"  # self-consistent on its own
    expect(check(keys, res.receipt, op, ch, expected={"prompt_tokens": real}), "PROMPT_BINDING")


def test_claiming_the_real_prompt_hash(engine, keys):
    from vgemma.protocol import io_chain, prompt_hash

    real = engine.tokenizer.apply_chat([{"role": "user", "content": PROMPT}])
    fake = list(real)
    fake[5:12] = [ord(c) + 16 for c in "IGNORE!"]
    res = engine.generate(prompt_tokens=fake, max_new_tokens=8)
    tr, com = engine.store.load(res.receipt["request_id"])

    def lie(rc):
        ph = prompt_hash(real)
        rc["prompt_hash"] = ph.hex()
        rc["io_chain_head"] = io_chain(
            ph, [(tr.tokens[tr.n_prompt + t], com.logits_hashes[t]) for t in range(tr.n_gen)]
        ).hex()

    receipt = resign(res.receipt, engine, lie)
    ch = make_challenge(receipt, 6, layers="full")
    op = engine.open(receipt["request_id"], ch)
    expect(check(keys, receipt, op, ch), "PROMPT_BINDING")
    op_real = mutate_opening(op, lambda idx, t: idx.__setitem__("prompt_tokens", real))
    expect(check(keys, receipt, op_real, ch), "PROMPT_BINDING")  # opened prompt != committed input tokens


def test_policy_must_match_client_request(engine, keys):
    r = Run(engine, policy=SamplingPolicy(temperature=0.5))
    asked = {"temperature": 1.0, "top_k": 64, "top_p": 0.95, "greedy": False, "max_new_tokens": 12, "thinking": False}
    assert check(keys, r.receipt, r.opening, r.challenge)["result"] == "PASS"
    expect(check(keys, r.receipt, r.opening, r.challenge, expected=asked), "MANIFEST_MISMATCH")
    expect(check(keys, r.receipt, r.opening, r.challenge, expected={**asked, "temperature": 0.5, "max_new_tokens": 99}),
           "MANIFEST_MISMATCH")  # fmt: skip


# 5. the challenge is the auditor's ------------------------------------------------------------------


def test_challenge_is_required(engine, keys):
    r = Run(engine)
    expect(check(keys, r.receipt, r.opening, None), "OPENING_SCHEMA")


def test_prover_chosen_empty_layer_set(make_engine, keys):
    bad = make_engine(tamper="weights")
    r = Run(bad)
    trace, com = bad.store.load(r.receipt["request_id"])
    own = r.with_(layers=[0])
    op = build_opening(trace, com, own, bad.profile, bad.embed_bits, bad.embed_tree)
    none = {**own, "audits": [{**a, "layers": []} for a in own["audits"]]}

    def empty(idx, t):
        idx["challenge"] = none

    expect(check(keys, r.receipt, mutate_opening(op, empty), none), "OPENING_SCHEMA")
    expect(check(keys, r.receipt, op, r.challenge), "OPENING_SCHEMA")  # not the auditor's challenge


# 6. single-logit boost within the old bf16 LM-head bound ------------------------------------------


def test_argmax_flip_by_sparse_logit_boost(make_engine, keys):
    def flip_top2(z, t):
        top = np.argsort(-z)[:2]
        z[top[1]] += (z[top[0]] - z[top[1]]) + 1e-3
        return z

    eng = wrap_logits(make_engine(), flip_top2)
    r = Run(eng, policy=SamplingPolicy(greedy=True), positions="random", layers="full")
    expect(check(keys, r.receipt, r.opening, r.challenge), "LMHEAD_BINDING")


def test_boost_below_former_bf16_bound_is_caught(make_engine, keys):
    """The bf16-output bound allowed one logit to move by about 3*2^-8*||z||; the f32 bound does not."""

    def nudge(z, t):
        z[np.argsort(-z)[1]] += 0.25 * 3 * 2.0**-8 * float(np.linalg.norm(z))
        return z

    eng = wrap_logits(make_engine(), nudge)
    r = Run(eng, policy=SamplingPolicy(greedy=True), positions="random", layers="0")
    expect(check(keys, r.receipt, r.opening, r.challenge), "LMHEAD_BINDING")


# 7. sampling randomness fixed by the client ---------------------------------------------------------


def test_client_nonce_honest(engine, keys):
    nonce = secrets.token_bytes(32)
    r = Run(engine, nonce=nonce)
    assert r.receipt["client_nonce"] == nonce.hex()
    prompt = engine.tokenizer.apply_chat([{"role": "user", "content": PROMPT}])
    v = check(keys, r.receipt, r.opening, r.challenge, expected={"client_nonce": nonce.hex(), "prompt_tokens": prompt})
    assert v["result"] == "PASS", dump(v)
    expect(check(keys, r.receipt, r.opening, r.challenge, expected={"client_nonce": nonce.hex()}), "PROMPT_BINDING")


def test_prover_ignores_client_nonce(engine, keys):
    nonce = secrets.token_bytes(32)
    r = Run(engine)  # sampled with the prover's own seed
    receipt = resign(r.receipt, engine, lambda rc: rc.__setitem__("client_nonce", nonce.hex()))
    expect(check(keys, receipt, r.opening, r.challenge), "SEED_COMMITMENT")
    expect(check(keys, r.receipt, r.opening, r.challenge, expected={"client_nonce": nonce.hex()}), "SEED_COMMITMENT")


def test_nonce_grinding_is_detected_by_expectation(engine, keys):
    nonce = secrets.token_bytes(32)
    other = Run(engine, nonce=secrets.token_bytes(32))  # the prover tried another nonce it liked better
    expect(check(keys, other.receipt, other.opening, other.challenge, expected={"client_nonce": nonce.hex()}),
           "SEED_COMMITMENT")  # fmt: skip


def test_receipt_schema_rejects_bad_nonce(engine, keys):
    r = Run(engine)
    receipt = copy.deepcopy(r.receipt)
    receipt["client_nonce"] = "zz"
    expect(check(keys, receipt, r.opening, r.challenge), "RECEIPT_SCHEMA")


def test_witness_postcap_binding(engine, keys):
    """The verifier recomputes the post-cap logits; the witness must commit to exactly those."""
    r = Run(engine, positions="random", layers="0")
    index, tensors = decode_opening(r.opening)
    p = r.full_positions[0]
    assert (
        index["positions"][str(p)]["witness"]["postcap"]
        == tensor_hash(softcap(tensors[f"p{p}/logits_precap"], 30.0)).hex()
    )
    assert encode_opening(index, tensors) == r.opening


# 8. second review: a residual stream forged at one layer boundary at every token ------------------


TARGET = [ord(c) + 16 for c in "FORGED-TEXT!"]


def p_forge_final(lm):
    """Replace the final norm's input so that the honest LM head picks the target token."""
    E, w = lm.lm_head.weight.detach().float(), lm.final_norm.weight.detach().float()
    state = {"t": 0}

    def pre(mod, args):
        x = args[0].clone()
        n = E[TARGET[state["t"] % len(TARGET)]] * w
        state["t"] += 1
        x[:, -1] = (n / n.pow(2).mean().sqrt()).to(x.dtype)
        return (x,)

    lm.final_norm.register_forward_pre_hook(pre)


def p_inject_layer5(lm):
    """Add a large target direction to layer 5's input and run layer 5 honestly."""
    E = lm.lm_head.weight.detach().float()
    state = {"t": 0}

    def pre(mod, args, kwargs):
        x = args[0].clone()
        x[:, -1] = (x[:, -1].float() + 300 * E[TARGET[state["t"] % len(TARGET)]]).to(x.dtype)
        state["t"] += 1
        return (x, *args[1:]), kwargs

    lm.text_model.layers[5].register_forward_pre_hook(pre, with_kwargs=True)


@pytest.mark.parametrize("patch,producer", [(p_forge_final, 5), (p_inject_layer5, 4)], ids=["r_final", "layer5_input"])
def test_forged_boundary_caught_by_decode_audits(make_engine, keys, patch, producer):
    eng = make_engine(patch=patch)
    r = Run(eng, layers="0,1,2", decode_layers=(producer,))  # full audits miss it; decode audits do not
    assert r.result.text.startswith("FORGED")
    v = check(keys, r.receipt, r.opening, r.challenge)
    expect(v, "BRIDGE_RESIDUAL")
    assert v["layer"] == producer
    missed = Run(eng, layers="0,1,2", decode="none")  # the earlier, shared-layer challenge
    assert check(keys, missed.receipt, missed.opening, missed.challenge)["result"] == "PASS"


def test_forged_boundary_detection_rate_of_default_challenge(make_engine, keys):
    from vgemma.auditor import forged_boundary_escape

    eng = make_engine(patch=p_forge_final)
    res = eng.generate(messages=[{"role": "user", "content": PROMPT}], max_new_tokens=12)
    escapes, caught = [], 0
    for _ in range(40):
        ch = make_challenge(res.receipt, 6)
        escapes.append(forged_boundary_escape(ch, 6))
        caught += check(keys, res.receipt, eng.open(res.receipt["request_id"], ch), ch)["result"] == "FAIL"
    assert max(escapes) < 0.05  # (1 - 3/6)^3 full audits x (1 - 1/6)^9 decode audits ~ 0.024
    assert caught >= 34, caught


# 9. second review: tokens at prompt positions the challenge does not open -----------------------


def test_substituted_prompt_tokens_at_unopened_positions(engine, keys):
    from vgemma.protocol import io_chain, prompt_hash

    real = engine.tokenizer.apply_chat([{"role": "user", "content": PROMPT}])
    fake = list(real)
    fake[2:9] = [ord(c) + 16 for c in "IGNORE!"]
    res = engine.generate(prompt_tokens=fake, max_new_tokens=8)
    tr, com = engine.store.load(res.receipt["request_id"])

    def lie(rc):
        ph = prompt_hash(real)
        rc["prompt_hash"] = ph.hex()
        rc["io_chain_head"] = io_chain(
            ph, [(tr.tokens[tr.n_prompt + t], com.logits_hashes[t]) for t in range(tr.n_gen)]
        ).hex()

    receipt = resign(res.receipt, engine, lie)
    from vgemma.auditor import challenge_from

    gen = list(range(tr.n_prompt - 1, tr.n_positions))
    ch = challenge_from(receipt["request_id"], gen[-1:], [0, 1, 2], attention=False)  # sliding layers, no attention
    op = mutate_opening(engine.open(receipt["request_id"], ch), lambda idx, t: idx.__setitem__("prompt_tokens", real))
    v = check(keys, receipt, op, ch, expected={"prompt_tokens": real})
    expect(v, "PROMPT_BINDING")
    assert v["position"] in range(2, 9)  # caught at a token-only position


# 10. third review: escape numbers and the remaining client fields --------------------------------


def p_fake_attention_last_layer(lm):
    """Replace the attention output `a` (o_proj input) at the last layer; o_proj and all else honest."""

    def pre(mod, args):
        x = args[0].clone()
        x[:, -1] = (x[:, -1].float() + 0.5 * x[:, -1].float().roll(1, dims=-1)).to(x.dtype)
        return (x,)

    lm.layer_module(5, "self_attn.o_proj").register_forward_pre_hook(pre)


def test_fake_attention_output_needs_attention_audits(make_engine, keys):
    from vgemma.auditor import challenge_from, forged_boundary_escape

    eng = make_engine(patch=p_fake_attention_last_layer)
    r = Run(eng, layers="0,1,2", decode_layers=(5,))
    v = check(keys, r.receipt, r.opening, r.challenge)
    assert v["result"] == "PASS"  # documented open item: layer 5 audited, but without the attention replay
    # priors over uniformly drawn layers: 3 attention audits of 3/6 layers; one-layer decode audits elsewhere
    sc = v["spot_check"]
    assert sc["fake_attention_escape"] == forged_boundary_escape(r.challenge, 6, attention_only=True) == 0.5**3
    assert abs(sc["forged_boundary_escape"] - 0.5**3 * (5 / 6) ** len(r.decode_positions)) < 1e-12
    gen = r.decode_positions
    ch = challenge_from(r.receipt["request_id"], gen[:1], [5], attention=True)
    expect(check(keys, r.receipt, eng.open(r.receipt["request_id"], ch), ch), "ATTN_REPLAY")
    with_attn = make_challenge(r.receipt, 6, layers="0", decode_layers=1, decode_attention=True)
    assert all(a["attention"] for a in with_attn["audits"])


def test_expected_attention_and_tokenizer_fields(engine, keys):
    r = Run(engine, request_id="r_rev0010")
    prompt = engine.tokenizer.apply_chat([{"role": "user", "content": PROMPT}])
    base = {"prompt_tokens": prompt, "attn_implementation": "sdpa", "tokenizer_hash": engine.tokenizer.tokenizer_hash,
            "chat_template_hash": engine.tokenizer.chat_template_hash}  # fmt: skip
    assert check(keys, r.receipt, r.opening, r.challenge, expected=base)["result"] == "PASS"
    for k, bad in (("attn_implementation", "eager"), ("tokenizer_hash", "ab" * 32), ("chat_template_hash", "cd" * 32)):
        expect(check(keys, r.receipt, r.opening, r.challenge, expected={**base, k: bad}), "MANIFEST_MISMATCH")
