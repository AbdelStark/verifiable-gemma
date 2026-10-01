"""Canonical functions against the transformers Gemma 4 modules, both layer types."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
import torch
from transformers.activations import ACT2FN
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4RMSNorm,
    Gemma4TextRotaryEmbedding,
    Gemma4TextScaledWordEmbedding,
    apply_rotary_pos_emb,
)

from vgemma import canon
from vgemma.model import to_numpy
from vgemma.prover.sampler import SamplingPolicy, sample, sample_step, uniform_for_step
from vgemma.tiny import tiny_config

U = canon.BF16_U


def bf16(x: torch.Tensor) -> np.ndarray:
    return to_numpy(x.to(torch.bfloat16))


def randn(*shape, scale=1.0, seed=0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16)


# -- bf16 arithmetic -------------------------------------------------------------------------------


def test_f32_to_bf16_matches_torch_rounding():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(200_000, generator=g) * torch.exp(torch.randn(200_000, generator=g) * 10)
    edge = torch.tensor(
        [
            0.0,
            -0.0,
            1.0,
            1.00390625,
            1.005859375,
            3.4e38,
            -3.4e38,
            1e-40,
            -1e-40,
            float("inf"),
            float("-inf"),
            65504.0,
            2.0**-133,
        ]
    )
    x = torch.cat([x, edge])
    ours = canon.f32_to_bf16(x.numpy())
    theirs = bf16(x)
    assert np.array_equal(ours, theirs)
    assert canon.f32_to_bf16(np.array([np.nan], dtype=np.float32))[0] & 0x7F80 == 0x7F80


@pytest.mark.parametrize("op", ["add", "mul"])
def test_bf16_add_mul_match_torch(op):
    a, b = randn(100_000, scale=4, seed=2), randn(100_000, scale=0.5, seed=3)
    ref = a + b if op == "add" else a * b
    fn = canon.bf16_add if op == "add" else canon.bf16_mul
    assert np.array_equal(fn(to_numpy(a), to_numpy(b)), to_numpy(ref))


@pytest.mark.parametrize("hidden,expected", [(64, 8.0), (3072, 55.5), (3840, 62.0), (5376, 73.5)])
def test_embed_scale_rounding(hidden, expected):
    emb = Gemma4TextScaledWordEmbedding(4, hidden, 0, embed_scale=hidden**0.5)
    torch_value = float(emb.embed_scale.to(torch.bfloat16))
    assert canon.embed_scale_bf16(hidden) == torch_value == expected


def test_scaled_embedding_replay_exact():
    emb = Gemma4TextScaledWordEmbedding(32, 64, 0, embed_scale=64**0.5).to(torch.bfloat16)
    with torch.no_grad():
        emb.weight.copy_(randn(32, 64, scale=0.5, seed=4))
    ids = torch.arange(32)
    out = to_numpy(emb(ids))
    scale = canon.f32_to_bf16(np.array([canon.embed_scale_bf16(64)], dtype=np.float32))
    assert np.array_equal(canon.bf16_mul(to_numpy(emb.weight), scale), out)


# -- norms and activations -------------------------------------------------------------------------


@pytest.mark.parametrize("dim", [16, 32, 64, 512])
@pytest.mark.parametrize("with_scale", [True, False])
def test_rmsnorm_gemma_matches_module(dim, with_scale):
    norm = Gemma4RMSNorm(dim, eps=1e-6, with_scale=with_scale)
    if with_scale:
        with torch.no_grad():
            norm.weight.copy_(torch.rand(dim) + 0.5)
    norm = norm.to(torch.bfloat16)
    x = randn(257, dim, scale=3, seed=dim)
    y = to_numpy(norm(x))
    w = canon.bf16_to_f64(to_numpy(norm.weight)) if with_scale else None
    ref = canon.rmsnorm_gemma(canon.bf16_to_f64(to_numpy(x)), w, 1e-6)
    rel = np.abs(canon.bf16_to_f64(y) - ref) / (np.abs(ref) + 1e-30)
    assert rel.max() <= U * (1 + 1e-3)  # one bf16 rounding of an f32 computation


def test_gelu_tanh_matches_module():
    act = ACT2FN["gelu_pytorch_tanh"]
    g = randn(100_000, scale=3, seed=7)
    u = randn(100_000, scale=1, seed=8)
    h = to_numpy(act(g) * u)  # the Gemma MLP gate: two bf16 roundings
    gf, uf = canon.bf16_to_f64(to_numpy(g)), canon.bf16_to_f64(to_numpy(u))
    ref = canon.gelu_tanh(gf) * uf
    err = np.abs(canon.bf16_to_f64(h) - ref)
    assert np.all(err <= 2 * U * np.abs(ref) * (1 + U) + 2.0**-24 * np.abs(gf * uf) + 2.0**-126)


def test_softcap_close_to_reference_and_deterministic():
    x = (np.random.default_rng(0).standard_normal(262_144) * 20).astype(np.float32)
    x[:4] = [0.0, -0.0, 1e-30, 500.0]
    y = canon.softcap(x, 30.0)
    ref = 30.0 * np.tanh(x.astype(np.float64) / 30.0)
    assert np.all(np.abs(y.astype(np.float64) - ref) <= 2.0**-23 * np.abs(ref) + 1e-38)
    # the same bits in a fresh interpreter (no hidden state, no libm dependence)
    code = (
        "import numpy as np, hashlib; from vgemma import canon;"
        "x=(np.random.default_rng(0).standard_normal(262144)*20).astype(np.float32); x[:4]=[0.0,-0.0,1e-30,500.0];"
        "print(hashlib.sha256(canon.softcap(x,30.0).tobytes()).hexdigest())"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    import hashlib

    assert out == hashlib.sha256(y.tobytes()).hexdigest()


def test_exp_f64_accuracy():
    x = np.linspace(-700, 700, 200_001)
    assert np.max(np.abs(canon.exp_f64(x) / np.exp(x) - 1)) < 1e-14
    assert canon.exp_f64(np.array([0.0]))[0] == 1.0


# -- RoPE -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("layer_type", ["sliding_attention", "full_attention"])
def test_rope_tables_match_transformers(layer_type):
    cfg = tiny_config()
    rot = Gemma4TextRotaryEmbedding(cfg)
    hd = cfg.per_layer_config[layer_type].head_dim
    pos = torch.tensor([[0, 1, 7, 8, 63, 1000, 4095]])
    cos_t, sin_t = rot(torch.zeros(1, dtype=torch.float32), pos, layer_type)
    cos, sin = canon.rope_cos_sin(cfg.rope_parameters[layer_type], hd, pos[0].numpy())
    assert np.max(np.abs(cos - cos_t[0].double().numpy())) < 2e-7
    assert np.max(np.abs(sin - sin_t[0].double().numpy())) < 2e-7
    if layer_type == "full_attention":  # partial rotary: 25 % of the pairs rotate, the rest pass through
        rot_pairs = int(0.25 * hd // 2)
        assert np.all(cos[:, rot_pairs : hd // 2] == 1.0) and np.all(sin[:, rot_pairs : hd // 2] == 0.0)
    # bf16 tables as the model sees them
    cos_b, _ = rot(torch.zeros(1, dtype=torch.bfloat16), pos, layer_type)
    assert np.array_equal(canon.f32_to_bf16(cos.astype(np.float32)), to_numpy(cos_b[0]))


@pytest.mark.parametrize("layer_type", ["sliding_attention", "full_attention"])
def test_apply_rope_bf16_bit_exact(layer_type):
    cfg = tiny_config()
    rot = Gemma4TextRotaryEmbedding(cfg)
    hd = cfg.per_layer_config[layer_type].head_dim
    n, heads = 40, 4
    pos = torch.arange(n)[None]
    x = randn(1, n, heads, hd, scale=1.0, seed=11)
    cos_b, sin_b = rot(x, pos, layer_type)
    ref = to_numpy(apply_rotary_pos_emb(x, cos_b, sin_b, unsqueeze_dim=2))[0]
    cos, sin = canon.rope_cos_sin(cfg.rope_parameters[layer_type], hd, np.arange(n))
    ours = canon.apply_rope_bf16(to_numpy(x)[0], cos, sin)
    assert np.array_equal(canon.f32_to_bf16(ours.astype(np.float32)), ref)


# -- sampler --------------------------------------------------------------------------------------


def _reference_sample(z: np.ndarray, policy: SamplingPolicy, u: float) -> int:
    z = z.astype(np.float64) / policy.temperature
    order = sorted(range(len(z)), key=lambda i: (-z[i], i))[: policy.top_k]
    e = np.exp(np.array([z[i] for i in order]) - z[order[0]])
    p = e / e.sum()
    cum, keep = 0.0, []
    for i, pi in zip(order, p, strict=True):
        keep.append(i)
        cum += pi
        if cum >= policy.top_p:
            break
    keep.sort()
    w = np.array([e[order.index(i)] for i in keep])
    cdf = np.cumsum(w) / w.sum()
    return keep[int(np.searchsorted(cdf, u, side="right"))]


def test_sampler_matches_reference_semantics():
    rng = np.random.default_rng(3)
    policy = SamplingPolicy(temperature=0.8, top_k=64, top_p=0.95)
    mismatches = 0
    for t in range(200):
        z = (rng.standard_normal(512) * 3).astype(np.float32)
        u = uniform_for_step(b"\x01" * 32, t)
        mismatches += sample(z, policy, u) != _reference_sample(z, policy, u)
    assert mismatches == 0


def test_sampler_greedy_ties_lowest_index():
    z = np.array([0.0, 5.0, 5.0, 1.0], dtype=np.float32)
    assert sample(z, SamplingPolicy(greedy=True), 0.99) == 1


def test_sampler_deterministic_across_processes():
    z = (np.random.default_rng(5).standard_normal(262_144) * 4).astype(np.float32)
    tok, wit = sample_step(z, SamplingPolicy(), b"\x07" * 32, 3)
    code = (
        "import numpy as np; from vgemma.prover.sampler import SamplingPolicy, sample_step;"
        "z=(np.random.default_rng(5).standard_normal(262144)*4).astype(np.float32);"
        "t,w=sample_step(z,SamplingPolicy(),b'\\x07'*32,3); print(t, repr(w['u']), w['postcap'])"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
    assert int(out[0]) == tok and float(out[1]) == wit["u"] and out[2] == wit["postcap"]


def test_descending_order_equals_stable_argsort():
    from vgemma.prover.sampler import descending_order

    rng = np.random.default_rng(9)
    for trial in range(200):
        n = int(rng.integers(2, 3000))
        z = rng.integers(-5, 5, n).astype(np.float64) if trial % 2 else rng.standard_normal(n)  # heavy ties
        z[rng.integers(0, n, 3)] = 0.0
        if trial % 7 == 0:
            z[:5] = -0.0
        for k in (1, 2, 5, 64, n - 1, n, 0):
            assert np.array_equal(descending_order(z, k), np.argsort(-z, kind="stable")[: k if k > 0 else n])
