"""Adversarial suite (tiny mode): every scenario must be rejected, for the right reason.

Four kinds of adversary, mirroring CommitLLM's scenario list (scripts/modal/tests/llama/
test_adversarial.py) plus the Gemma 4 specific ones:

* tamper modes of the serving engine (the demo modes and more);
* post-commit edits of the opening or the receipt (bit flips, swaps, splices, wrong seed, ...);
* a prover that commits to a modified trace, so the opening is consistent with the receipt and
  only the semantic checks can catch it;
* a prover serving a consistently modified model (wrong activation, norm form, scaling, ...).
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from helpers import Run, check, dump, mutate_opening, recommit, resign
from vgemma.canon import bf16_to_f32, f32_to_bf16, softcap, tensor_hash
from vgemma.profile import GLOBAL_NAMES, SLIDING_NAMES
from vgemma.protocol import decode_opening, logits_hash, tensor_key
from vgemma.prover.tamper import EXPECTED_CODES, TAMPER_MODES, Tamper


def expect(v, code):
    assert v["result"] == "FAIL", f"expected {code}, got PASS"
    assert v["reason"] == code, f"expected {code}, got {dump(v)}"


@pytest.fixture(scope="module")
def run(engine):
    return Run(engine, max_new_tokens=12, positions="edges", layers="full", request_id="r_adv0001")


@pytest.fixture(scope="module")
def run_b(engine):
    return Run(engine, max_new_tokens=12, positions="edges", layers="full", request_id="r_adv0002")


def first_pos(run):
    return run.full_positions[0]


# ---------------------------------------------------------------------------
# 1. tamper modes of the serving engine
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", TAMPER_MODES)
def test_tamper_modes(make_engine, keys, mode):
    eng = make_engine(tamper=mode)
    r = Run(eng, max_new_tokens=12, positions="all-gen", layers="full")
    expect(check(keys, r.receipt, r.opening, r.challenge), EXPECTED_CODES[mode])


@pytest.mark.parametrize("layer", range(6))
def test_weights_tamper_every_layer(make_engine, keys, layer):
    eng = make_engine(tamper=Tamper("weights", layer=layer))
    r = Run(eng, positions="edges", layers="full")
    v = check(keys, r.receipt, r.opening, r.challenge)
    expect(v, "FREIVALDS_WDOWN")
    assert v["layer"] == layer and v["detail"]["deviation"] > v["detail"]["tolerance"]


def test_weights_tamper_missed_when_layer_not_challenged(make_engine, keys):
    """Routine audits only see the layers they open: the claim is per challenged layer."""
    eng = make_engine(tamper=Tamper("weights", layer=3))
    r = Run(eng, positions="edges", layers="0,1,2")
    assert check(keys, r.receipt, r.opening, r.challenge)["result"] == "PASS"


# ---------------------------------------------------------------------------
# 2. post-commit edits of the opening
# ---------------------------------------------------------------------------


def flip(arr: np.ndarray, i: int = 0) -> None:
    flat = arr.reshape(-1).view(np.uint32 if arr.dtype == np.float32 else arr.dtype)
    flat[i] ^= 1


@pytest.mark.parametrize("name", SLIDING_NAMES)
def test_bitflip_sliding_layer_tensor(run, keys, name):
    p = first_pos(run)
    op = mutate_opening(run.opening, lambda idx, t: flip(t[tensor_key(p, 0, name)]))
    expect(check(keys, run.receipt, op, run.challenge), "MERKLE_POSITION")


@pytest.mark.parametrize("name", GLOBAL_NAMES)
def test_bitflip_global_layer_tensor(run, keys, name):
    p = first_pos(run)
    op = mutate_opening(run.opening, lambda idx, t: flip(t[tensor_key(p, 5, name)]))
    expect(check(keys, run.receipt, op, run.challenge), "MERKLE_POSITION")


@pytest.mark.parametrize("name", ["r_final", "h_final"])
def test_bitflip_final_group(run, keys, name):
    p = first_pos(run)
    op = mutate_opening(run.opening, lambda idx, t: flip(t[tensor_key(p, 6, name)]))
    expect(check(keys, run.receipt, op, run.challenge), "MERKLE_POSITION")


def test_bitflip_logits(run, keys):
    p = first_pos(run)
    op = mutate_opening(run.opening, lambda idx, t: flip(t[f"p{p}/logits_precap"], 7))
    expect(check(keys, run.receipt, op, run.challenge), "MERKLE_POSITION")


def test_bitflip_decode_only_logits(run, keys):
    d = run.decode_positions[0]
    op = mutate_opening(run.opening, lambda idx, t: flip(t[f"p{d}/logits_precap"], 3))
    expect(check(keys, run.receipt, op, run.challenge), "MERKLE_POSITION")


def test_witness_postcap_hash_edit(run, keys):
    p = first_pos(run)
    op = mutate_opening(
        run.opening, lambda idx, t: idx["positions"][str(p)]["witness"].__setitem__("postcap", "ab" * 32)
    )
    expect(check(keys, run.receipt, op, run.challenge), "MERKLE_POSITION")


def test_bitflip_embedding_row(run, keys):
    p = first_pos(run)
    op = mutate_opening(run.opening, lambda idx, t: flip(t[f"p{p}/embed_row"]))
    expect(check(keys, run.receipt, op, run.challenge), "EMBEDDING")


def test_bitflip_provenance_row(run, keys):
    p = first_pos(run)
    j = p - 3  # inside the sliding window of p, not itself challenged
    assert str(j) in decode_opening(run.opening)[0]["positions"]
    op = mutate_opening(run.opening, lambda idx, t: flip(t[tensor_key(j, 0, "k_n")]))
    expect(check(keys, run.receipt, op, run.challenge), "KV_PROVENANCE")


def test_missing_provenance_row(run, keys):
    p = first_pos(run)
    j = p - 2

    def drop(idx, t):
        g = idx["positions"][str(j)]["layers"]["0"]["tensors"]
        g["v_n"] = tensor_hash(t.pop(tensor_key(j, 0, "v_n"))).hex()

    op = mutate_opening(run.opening, drop)
    expect(check(keys, run.receipt, op, run.challenge), "KV_PROVENANCE")


def test_swapped_layers(run, keys):
    p = first_pos(run)

    def swap(idx, t):
        e = idx["positions"][str(p)]["layers"]
        e["0"], e["1"] = e["1"], e["0"]
        for name in SLIDING_NAMES:
            a, b = tensor_key(p, 0, name), tensor_key(p, 1, name)
            t[a], t[b] = t[b], t[a]

    expect(check(keys, run.receipt, mutate_opening(run.opening, swap), run.challenge), "MERKLE_POSITION")


def test_swapped_positions(run, keys):
    p, q = run.full_positions[:2]

    def swap(idx, t):
        e = idx["positions"]
        e[str(p)], e[str(q)] = e[str(q)], e[str(p)]
        moved = {}
        for k in list(t):
            for a, b in ((p, q), (q, p)):
                if k.startswith(f"p{a}/"):
                    moved[f"p{b}/" + k[len(f"p{a}/") :]] = t.pop(k)
        t.update(moved)

    expect(check(keys, run.receipt, mutate_opening(run.opening, swap), run.challenge), "MERKLE_POSITION")


def test_wrong_merkle_sibling(run, keys):
    p = first_pos(run)

    def bad(idx, t):
        pr = idx["positions"][str(p)]["proof"]
        pr[1] = ("0" if pr[1][0] != "0" else "1") + pr[1][1:]

    expect(check(keys, run.receipt, mutate_opening(run.opening, bad), run.challenge), "MERKLE_POSITION")


def test_truncated_merkle_path(run, keys):
    p = first_pos(run)
    op = mutate_opening(run.opening, lambda idx, t: idx["positions"][str(p)]["proof"].pop())
    expect(check(keys, run.receipt, op, run.challenge), "MERKLE_POSITION")


def test_token_index_shift(run, keys):
    p = first_pos(run)
    op = mutate_opening(run.opening, lambda idx, t: idx["positions"][str(p)].__setitem__("input_token", 99))
    expect(check(keys, run.receipt, op, run.challenge), "MERKLE_POSITION")


def test_sampled_token_flip(run, keys):
    p = first_pos(run)
    e_tok = lambda idx: idx["positions"][str(p)]  # noqa: E731
    op = mutate_opening(run.opening, lambda idx, t: e_tok(idx).__setitem__("out_token", e_tok(idx)["out_token"] ^ 1))
    expect(check(keys, run.receipt, op, run.challenge), "MERKLE_POSITION")


def test_witness_u_edit(run, keys):
    p = first_pos(run)
    op = mutate_opening(run.opening, lambda idx, t: idx["positions"][str(p)]["witness"].__setitem__("u", 0.5))
    expect(check(keys, run.receipt, op, run.challenge), "MERKLE_POSITION")


def test_wrong_seed(run, keys):
    op = mutate_opening(run.opening, lambda idx, t: idx.__setitem__("seed", "00" * 32))
    expect(check(keys, run.receipt, op, run.challenge), "SEED_COMMITMENT")


def test_prompt_append_token(run, keys):
    op = mutate_opening(run.opening, lambda idx, t: idx["prompt_tokens"].append(20))
    expect(check(keys, run.receipt, op, run.challenge), "PROMPT_BINDING")


def test_prompt_token_changed(run, keys):
    op = mutate_opening(run.opening, lambda idx, t: idx["prompt_tokens"].__setitem__(5, 77))
    expect(check(keys, run.receipt, op, run.challenge), "PROMPT_BINDING")


def test_io_transcript_token_flip(run, keys):
    op = mutate_opening(run.opening, lambda idx, t: idx["io_transcript"][2].__setitem__(0, 300))
    expect(check(keys, run.receipt, op, run.challenge), "IO_CHAIN")


def test_io_transcript_truncated(run, keys):
    op = mutate_opening(run.opening, lambda idx, t: idx["io_transcript"].pop())
    expect(check(keys, run.receipt, op, run.challenge), "IO_CHAIN")


def test_challenged_layer_withheld(run, keys):
    p = first_pos(run)

    def withhold(idx, t):
        idx["positions"][str(p)]["layers"]["2"] = {"leaf": "00" * 32}
        for k in [k for k in t if k.startswith(f"p{p}/l2/")]:
            t.pop(k)

    expect(check(keys, run.receipt, mutate_opening(run.opening, withhold), run.challenge), "OPENING_SCHEMA")


def test_removed_v_norm_name(run, keys):
    p = first_pos(run)

    def drop(idx, t):
        idx["positions"][str(p)]["layers"]["5"]["tensors"].pop("v_n")
        t.pop(tensor_key(p, 5, "v_n"))

    expect(check(keys, run.receipt, mutate_opening(run.opening, drop), run.challenge), "WIRING")


def test_v_proj_on_global_layer(run, keys):
    p = first_pos(run)

    def add_v(idx, t):
        idx["positions"][str(p)]["layers"]["5"]["tensors"]["v"] = "open"
        t[tensor_key(p, 5, "v")] = t[tensor_key(p, 5, "k")].copy()

    expect(check(keys, run.receipt, mutate_opening(run.opening, add_v), run.challenge), "WIRING")


def test_wrong_head_shape(run, keys):
    p = first_pos(run)
    op = mutate_opening(
        run.opening, lambda idx, t: t.__setitem__(tensor_key(p, 5, "q_n"), t[tensor_key(p, 5, "q_n")].reshape(8, 16))
    )
    expect(check(keys, run.receipt, op, run.challenge), "WIRING")


def test_challenge_mismatch(run, keys):
    other = run.with_(layers=[0, 1])
    expect(check(keys, run.receipt, run.opening, other), "OPENING_SCHEMA")


def test_opening_for_another_request(run, run_b, keys):
    expect(check(keys, run.receipt, run_b.opening, run.challenge), "OPENING_SCHEMA")


def test_cross_request_position_splice(run, run_b, keys):
    p = first_pos(run)
    assert p in run_b.full_positions
    idx_b, t_b = decode_opening(run_b.opening)

    def splice(idx, t):
        idx["positions"][str(p)] = idx_b["positions"][str(p)]
        for k in t_b:
            if k.startswith(f"p{p}/"):
                t[k] = t_b[k]

    expect(check(keys, run.receipt, mutate_opening(run.opening, splice), run.challenge), "MERKLE_POSITION")


def test_embedding_proof_for_wrong_token(run, keys):
    p = first_pos(run)
    idx0, _ = decode_opening(run.opening)
    tok = idx0["positions"][str(p)]["input_token"]
    other = (tok + 1) % 512

    def swap_row(idx, t):
        t[f"p{p}/embed_row"] = run.engine.embed_bits[other].copy()
        idx["positions"][str(p)]["embedding"]["proof"] = [h.hex() for h in run.engine.embed_tree.proof(other)]

    expect(check(keys, run.receipt, mutate_opening(run.opening, swap_row), run.challenge), "EMBEDDING")


def test_logits_withheld(run, keys):
    p = first_pos(run)

    def hide(idx, t):
        idx["positions"][str(p)]["logits"] = logits_hash(t.pop(f"p{p}/logits_precap")).hex()

    expect(check(keys, run.receipt, mutate_opening(run.opening, hide), run.challenge), "OPENING_SCHEMA")


def test_malformed_opening(run, keys):
    expect(check(keys, run.receipt, b"not an opening", run.challenge), "OPENING_SCHEMA")
    expect(check(keys, run.receipt, run.opening[:-100], run.challenge), "OPENING_SCHEMA")


# ---------------------------------------------------------------------------
# 3. receipt edits (third party: breaks the signature; prover: re-signed)
# ---------------------------------------------------------------------------


def test_manifest_edit_after_signing(run, keys):
    r = copy.deepcopy(run.receipt)
    r["manifest"]["temperature"] = 0.7
    expect(check(keys, r, run.opening, run.challenge), "RECEIPT_SIGNATURE")


def test_signature_flip(run, keys):
    r = copy.deepcopy(run.receipt)
    sig = r["prover"]["signature"]
    r["prover"]["signature"] = sig[:-1] + ("0" if sig[-1] != "0" else "1")
    expect(check(keys, r, run.opening, run.challenge), "RECEIPT_SIGNATURE")


def test_unpinned_prover(run, keys):
    expect(check(keys, run.receipt, run.opening, run.challenge, prover_id="ed25519:" + "ab" * 32), "RECEIPT_SIGNATURE")


def test_receipt_schema(run, keys):
    r = copy.deepcopy(run.receipt)
    r["version"] = 2
    expect(check(keys, r, run.opening, run.challenge), "RECEIPT_SCHEMA")
    r = copy.deepcopy(run.receipt)
    del r["manifest"]["top_k"]
    expect(check(keys, r, run.opening, run.challenge), "RECEIPT_SCHEMA")


def _flip_hex(h: str) -> str:
    return ("0" if h[0] != "0" else "1") + h[1:]


RESIGNED = [
    ("temperature", lambda r: r["manifest"].__setitem__("temperature", 0.6), "DECODE_SAMPLING"),
    ("top_k_1", lambda r: r["manifest"].__setitem__("top_k", 1), "DECODE_SAMPLING"),
    ("greedy", lambda r: r["manifest"].__setitem__("greedy", True), "DECODE_SAMPLING"),
    ("softcap_value", lambda r: r["manifest"].__setitem__("final_logit_softcapping", 50.0), "MANIFEST_MISMATCH"),
    ("embed_scale", lambda r: r["manifest"].__setitem__("embed_scale_bf16", 8.0625), "MANIFEST_MISMATCH"),
    ("eos_ids", lambda r: r["manifest"].__setitem__("eos_token_ids", [1]), "MANIFEST_MISMATCH"),
    ("tokenizer", lambda r: r["manifest"].__setitem__("tokenizer_hash", "ab" * 32), "MANIFEST_MISMATCH"),
    ("rope_full", lambda r: r["manifest"].__setitem__("rope_hash", "cd" * 32), "WIRING"),
    ("window", lambda r: r["manifest"].__setitem__("sliding_window", 16), "WIRING"),
    ("k_eq_v", lambda r: r["manifest"].__setitem__("attention_k_eq_v", False), "WIRING"),
    ("speculative", lambda r: r["manifest"].__setitem__("speculative", "eagle"), "MANIFEST_UNSUPPORTED"),
    ("prefix_cache", lambda r: r["manifest"].__setitem__("prefix_caching", True), "MANIFEST_UNSUPPORTED"),
    ("dtype", lambda r: r["manifest"].__setitem__("dtype", "float16"), "MANIFEST_UNSUPPORTED"),
    (
        "attn_impl",
        lambda r: r["manifest"].__setitem__("attn_implementation", "flash_attention_2"),
        "MANIFEST_UNSUPPORTED",
    ),
    ("n_gen_inflate", lambda r: r.__setitem__("n_gen", r["n_gen"] + 1), "IO_CHAIN"),
    ("n_prompt_inflate", lambda r: r.__setitem__("n_prompt", r["n_prompt"] + 1), "PROMPT_BINDING"),
    ("trace_root", lambda r: r.__setitem__("trace_root", _flip_hex(r["trace_root"])), "MERKLE_POSITION"),
    ("io_chain_head", lambda r: r.__setitem__("io_chain_head", _flip_hex(r["io_chain_head"])), "IO_CHAIN"),
    ("seed_commitment", lambda r: r.__setitem__("seed_commitment", _flip_hex(r["seed_commitment"])), "SEED_COMMITMENT"),
    ("prompt_hash", lambda r: r.__setitem__("prompt_hash", _flip_hex(r["prompt_hash"])), "PROMPT_BINDING"),
    (
        "weights_root",
        lambda r: r["model"].__setitem__("weights_root", _flip_hex(r["model"]["weights_root"])),
        "WEIGHTS_ROOT",
    ),
    (
        "config_hash",
        lambda r: r["model"].__setitem__("config_hash", _flip_hex(r["model"]["config_hash"])),
        "CONFIG_HASH",
    ),
    ("max_new_tokens", lambda r: r["manifest"].__setitem__("max_new_tokens", 64), "IO_CHAIN"),
]


@pytest.mark.parametrize("name,edit,code", RESIGNED, ids=[x[0] for x in RESIGNED])
def test_resigned_receipt_edits(engine, keys, name, edit, code):
    r = Run(engine, positions="all-gen", layers="5", request_id="r_adv0003")
    expect(check(keys, resign(r.receipt, engine, edit), r.opening, r.challenge), code)


# ---------------------------------------------------------------------------
# 4. consistent malicious commitments (opening matches the receipt)
# ---------------------------------------------------------------------------


def bump(trace, layer: int, name: str, rel: float = 0.25) -> None:
    """Shift one element of a captured tensor at every position by ``rel * ||row||``."""
    arr = trace.layers[layer][name]
    flat = arr.reshape(arr.shape[0], -1)
    vals = bf16_to_f32(flat).astype(np.float64)
    vals[:, 0] += rel * np.linalg.norm(vals, axis=1) + 1.0
    flat[:] = f32_to_bf16(vals.astype(np.float32))


SHELL_CASES = [
    (0, "q", "FREIVALDS_WQ"),
    (0, "x_attn", "FREIVALDS_WQ"),
    (0, "k", "FREIVALDS_WK"),
    (0, "v", "FREIVALDS_WV"),
    (0, "o", "FREIVALDS_WO"),
    (0, "a", "FREIVALDS_WO"),
    (0, "g", "FREIVALDS_WGATE"),
    (0, "x_ffn", "FREIVALDS_WGATE"),
    (0, "u", "FREIVALDS_WUP"),
    (0, "d", "FREIVALDS_WDOWN"),
    (0, "h", "FREIVALDS_WDOWN"),
    (5, "q", "FREIVALDS_WQ"),
    (5, "k", "FREIVALDS_WK"),
    (5, "d", "FREIVALDS_WDOWN"),
]
BRIDGE_CASES = [
    (1, "q_n", "BRIDGE_NORM_Q"),
    (1, "k_n", "BRIDGE_NORM_K"),
    (1, "v_n", "BRIDGE_NORM_V"),
    (5, "v_n", "KV_SHARED"),
    (1, "o_n", "BRIDGE_NORM_POST_ATTN"),
    (1, "r_mid", "BRIDGE_RESIDUAL"),
    (1, "d_n", "BRIDGE_NORM_POST_FFN"),
    (6, "h_final", "BRIDGE_NORM_FINAL"),
    (6, "r_final", "BRIDGE_RESIDUAL"),
    (0, "r_in", "EMBEDDING"),
]


@pytest.mark.parametrize("layer,name,code", SHELL_CASES + BRIDGE_CASES, ids=lambda x: str(x))
def test_consistent_trace_tamper(engine, keys, run, layer, name, code):
    receipt, opening, ch = recommit(engine, run, lambda tr: bump(tr, layer, name))
    expect(check(keys, receipt, opening, ch), code)


def test_consistent_residual_stream_tamper(engine, keys, run):
    """A changed layer input is caught by the previous layer's residual replay, or, when only the
    layer itself is challenged, by its input norm."""
    receipt, opening, ch = recommit(engine, run, lambda tr: bump(tr, 2, "r_in"))
    v = check(keys, receipt, opening, ch)
    expect(v, "BRIDGE_RESIDUAL")
    assert v["layer"] == 1
    receipt, opening, ch = recommit(engine, run, lambda tr: bump(tr, 2, "r_in"), challenge=run.with_(layers=[2]))
    expect(check(keys, receipt, opening, ch), "BRIDGE_NORM_INPUT")


def test_consistent_fake_attention_output_caught_at_challenged_position(engine, keys, run):
    """Changing `a` and re-deriving o = W_o a is out of reach without weights here, but a fake `a`
    alone already fails Wo; with o consistent the attention replay is what remains (audited)."""
    receipt, opening, ch = recommit(engine, run, lambda tr: bump(tr, 1, "a", rel=0.02))
    v = check(keys, receipt, opening, ch)
    assert v["result"] == "FAIL" and v["reason"] in ("FREIVALDS_WO", "ATTN_REPLAY"), dump(v)


def test_consistent_shell_layer_swap(engine, keys, run):
    def swap(tr):
        tr.layers[1], tr.layers[2] = tr.layers[2], tr.layers[1]

    receipt, opening, ch = recommit(engine, run, swap)
    v = check(keys, receipt, opening, ch)
    expect(v, "BRIDGE_RESIDUAL")  # layer 0's output no longer feeds "layer 1"
    receipt, opening, ch = recommit(engine, run, swap, challenge=run.with_(layers=[1, 2]))
    expect(check(keys, receipt, opening, ch), "FREIVALDS_WQ")  # layer-2 activations against layer-1 weights
    swap01 = lambda tr: tr.layers.update({0: tr.layers[1], 1: tr.layers[0]})  # noqa: E731
    receipt, opening, ch = recommit(engine, run, swap01)
    expect(check(keys, receipt, opening, ch), "EMBEDDING")


def test_consistent_cross_request_layer_splice(engine, keys, run, run_b):
    def splice(tr):
        n = min(tr.n_positions, run_b.trace.n_positions)
        for name, arr in tr.layers[2].items():
            arr[:n] = run_b.trace.layers[2][name][:n]

    receipt, opening, ch = recommit(engine, run, splice)
    expect(check(keys, receipt, opening, ch), "BRIDGE_RESIDUAL")


def test_consistent_logits_tamper(engine, keys, run):
    def edit(tr):
        tr.logits_precap[:, 3] += 5.0
        tr.logits_postcap[:] = np.stack([softcap(x, 30.0) for x in tr.logits_precap])
        for t, w in enumerate(tr.witnesses):
            w["postcap"] = tensor_hash(tr.logits_postcap[t]).hex()

    receipt, opening, ch = recommit(engine, run, edit)
    expect(check(keys, receipt, opening, ch), "LMHEAD_BINDING")


def test_consistent_token_substitution(engine, keys, run):
    def edit(tr):
        i = tr.n_prompt + 1
        tr.tokens[i] = (tr.tokens[i] + 1) % 512

    ch = run.with_(positions=[run.trace.n_prompt], decode_positions=[])
    receipt, opening, ch = recommit(engine, run, edit, challenge=ch)
    expect(check(keys, receipt, opening, ch), "DECODE_SAMPLING")
    # with decode audits on every token, the next position's input-token embedding catches it first
    receipt, opening, ch = recommit(engine, run, edit, challenge=run.with_(positions=[run.trace.n_prompt]))
    expect(check(keys, receipt, opening, ch), "EMBEDDING")


def test_consistent_witness_u(engine, keys, run):
    ch = run.with_(positions=[run.trace.n_prompt - 1])
    receipt, opening, ch = recommit(engine, run, lambda tr: tr.witnesses[0].__setitem__("u", 0.123), challenge=ch)
    expect(check(keys, receipt, opening, ch), "DECODE_SAMPLING")


def test_consistent_truncation(engine, keys, run):
    def cut(tr):
        keep = tr.n_gen - 3
        tr.tokens = tr.tokens[: tr.n_prompt + keep]
        tr.n_gen = keep
        n = tr.n_prompt + keep - 1
        tr.layers = {lay: {k: a[:n] for k, a in g.items()} for lay, g in tr.layers.items()}
        tr.logits_precap, tr.logits_postcap = tr.logits_precap[:keep], tr.logits_postcap[:keep]
        tr.witnesses = tr.witnesses[:keep]

    if run.trace.tokens[-1] in run.engine.lm.eos_token_ids:
        pytest.skip("run ended at EOS")
    ch = run.with_(positions=[run.trace.n_prompt - 1], decode_positions=[])
    receipt, opening, ch = recommit(engine, run, cut, challenge=ch)
    expect(check(keys, receipt, opening, ch), "IO_CHAIN")


# ---------------------------------------------------------------------------
# 5. a prover serving a consistently modified model
# ---------------------------------------------------------------------------


def _layers(lm):
    return lm.text_model.layers


def p_silu(lm):
    for layer in _layers(lm):
        layer.mlp.act_fn = torch.nn.SiLU()


def p_one_plus_w(lm):
    for layer in _layers(lm):
        m = layer.input_layernorm
        m.forward = lambda x, m=m: (m._norm(x.float()) * (1.0 + m.weight.float())).type_as(x)


def p_no_layer_scalar(lm):
    for layer in _layers(lm):
        layer.layer_scalar.fill_(1.0)


def p_embed_scale(lm):
    lm.text_model.embed_tokens.embed_scale.fill_(9.0)


def p_final_norm(lm):
    with torch.no_grad():
        lm.final_norm.weight.mul_(1.1)


def p_untied_head(lm):
    w = lm.lm_head.weight.detach().clone()
    w += torch.randn_like(w.float()).to(w.dtype) * 0.1
    lm.lm_head.weight = torch.nn.Parameter(w)


def p_window(lm):
    lm.text_model.config.sliding_window = 16
    for layer in _layers(lm):
        if layer.self_attn.sliding_window:
            layer.self_attn.sliding_window = 16


def p_scaling(lm):
    for layer in _layers(lm):
        layer.self_attn.scaling = layer.self_attn.head_dim**-0.5


def p_global_v_norm(lm):
    lm.text_model.layers[5].self_attn.v_norm.forward = lambda x: x


def p_no_q_norm(lm):
    for layer in _layers(lm):
        layer.self_attn.q_norm.forward = lambda x: x


def p_delta(suffix: str, layer: int):
    def patch(lm):
        w = lm.layer_module(layer, suffix).weight
        g = torch.Generator().manual_seed(7)
        a = torch.randn(w.shape[0], 8, generator=g, dtype=torch.float64)
        b = torch.randn(w.shape[1], 8, generator=g, dtype=torch.float64)
        d = a @ b.T
        d *= 0.03 * w.double().norm() / d.norm()
        with torch.no_grad():
            w.copy_((w.double() + d).to(w.dtype))

    return patch


MODEL_CASES = [
    ("silu_instead_of_gelu", p_silu, "full", "BRIDGE_GELU"),
    ("llama_style_norm", p_one_plus_w, "full", "BRIDGE_NORM_INPUT"),
    ("layer_scalar_dropped", p_no_layer_scalar, "full", "BRIDGE_RESIDUAL"),
    ("embed_scale", p_embed_scale, "full", "EMBEDDING"),
    ("final_norm_weight", p_final_norm, "full", "BRIDGE_NORM_FINAL"),
    ("untied_lm_head", p_untied_head, "full", "LMHEAD_BINDING"),
    ("sliding_window_16", p_window, "full", "ATTN_REPLAY"),
    ("attention_scaling", p_scaling, "full", "ATTN_REPLAY"),
    ("global_v_norm_skipped", p_global_v_norm, "5", "KV_SHARED"),
    ("q_norm_skipped", p_no_q_norm, "full", "BRIDGE_NORM_Q"),
    ("delta_q_proj_l2", p_delta("self_attn.q_proj", 2), "full", "FREIVALDS_WQ"),
    ("delta_k_proj_l5", p_delta("self_attn.k_proj", 5), "full", "FREIVALDS_WK"),
    ("delta_v_proj_l1", p_delta("self_attn.v_proj", 1), "full", "FREIVALDS_WV"),
    ("delta_o_proj_l4", p_delta("self_attn.o_proj", 4), "full", "FREIVALDS_WO"),
    ("delta_gate_l0", p_delta("mlp.gate_proj", 0), "full", "FREIVALDS_WGATE"),
    ("delta_up_l3", p_delta("mlp.up_proj", 3), "full", "FREIVALDS_WUP"),
]


@pytest.mark.parametrize("name,patch,layers,code", MODEL_CASES, ids=[c[0] for c in MODEL_CASES])
def test_modified_model(make_engine, keys, name, patch, layers, code):
    eng = make_engine(patch=patch)
    r = Run(eng, positions="all-gen", layers=layers)
    expect(check(keys, r.receipt, r.opening, r.challenge), code)
