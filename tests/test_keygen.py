"""Keygen on the tiny checkpoint: families per layer type, the Freivalds identity, fail-closed configs."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from vgemma.canon import bf16_to_f64
from vgemma.keygen import keygen
from vgemma.model import iter_checkpoint_tensors, load_model
from vgemma.profile import GemmaProfile, UnsupportedModel
from vgemma.tiny import tiny_config


def test_families_per_layer_type(keys):
    key, public = keys
    prof = key.profile
    assert prof.layer_types[-1] == "full_attention"
    for layer in range(prof.num_layers):
        names = set(key._npz.files)
        has_wv = f"v.{layer}.wv" in names
        assert has_wv == (not prof.is_global(layer)), "global layers must have no Wv family"
        for fam in prof.families(layer):
            out_f, in_f = prof.family_shape(layer, fam)
            r, v = key.r(layer, fam), key.v(layer, fam)
            assert r.dtype == np.int8 and r.shape == (key.k, out_f) and set(np.unique(r)) == {-1, 1}
            assert v.dtype == np.float64 and v.shape == (key.k, in_f)
    assert key.r_lm.shape == (key.k, prof.vocab_size) and key.v_lm.shape == (key.k, prof.hidden_size)
    assert public["keygen"]["key_bytes"] == key.size_bytes > 0
    assert public["eos_token_ids"] == [1, 4]


def test_freivalds_identity_holds_exactly(keys, tiny_dir):
    key, _ = keys
    tensors = dict(iter_checkpoint_tensors(tiny_dir))
    w = bf16_to_f64(tensors["model.layers.5.self_attn.q_proj.weight"])
    x = np.random.default_rng(0).standard_normal(w.shape[1])
    lhs = key.r(5, "wq").astype(np.float64) @ (w @ x)
    rhs = key.v(5, "wq") @ x
    assert np.allclose(lhs, rhs, rtol=0, atol=1e-10)


def test_key_vectors_are_secret_and_fresh(tiny_dir, tmp_path, keys):
    key, _ = keys
    keygen(str(tiny_dir), tmp_path / "k2", log=lambda *_: None)
    other = np.load(tmp_path / "k2" / "key.npz")
    assert not np.array_equal(other["r.0.wq"], key.r(0, "wq"))
    pub2 = json.loads((tmp_path / "k2" / "public.json").read_text())
    assert pub2["weights_root"] == keys[1]["weights_root"]  # public part is deterministic


def test_layer_scalars_in_key(keys, tiny_dir):
    key, _ = keys
    tensors = dict(iter_checkpoint_tensors(tiny_dir))
    for layer in range(key.profile.num_layers):
        assert np.array_equal(key.layer_scalar_bits(layer), tensors[f"model.layers.{layer}.layer_scalar"])


@pytest.mark.parametrize(
    "override,match",
    [
        ({"hidden_size_per_layer_input": 16}, "per-layer input embeddings"),
        ({"num_kv_shared_layers": 2}, "KV-shared"),
        ({"use_double_wide_mlp": True}, "double_wide"),
        ({"attention_k_eq_v": False}, "attention_k_eq_v"),
        ({"use_bidirectional_attention": "all"}, "bidirectional"),
        ({"hidden_activation": "silu"}, "gelu_pytorch_tanh"),
        ({"final_logit_softcapping": None}, "final_logit_softcapping"),
        ({"attention_bias": True}, "attention_bias"),
    ],
)
def test_fail_closed_configs(override, match):
    with pytest.raises(UnsupportedModel, match=match):
        GemmaProfile.from_hf_config(tiny_config(**override))


def test_fail_closed_moe():
    cfg = tiny_config(enable_moe_block=True, num_experts=4, top_k_experts=2, moe_intermediate_size=32)
    with pytest.raises(UnsupportedModel, match="Mixture-of-experts"):
        GemmaProfile.from_hf_config(cfg)


def test_fail_closed_unknown_layer_type():
    cfg = tiny_config()
    cfg.layer_types = ["sliding_attention"] * 5 + ["chunked_attention"]
    with pytest.raises(UnsupportedModel, match="unknown layer_types"):
        GemmaProfile.from_hf_config(cfg)


def test_keygen_and_serve_refuse_unsupported_checkpoint(tiny_dir, tmp_path):
    bad = tmp_path / "bad"
    shutil.copytree(tiny_dir, bad)
    cfg = json.loads((bad / "config.json").read_text())
    cfg["attention_k_eq_v"] = False
    (bad / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(UnsupportedModel):
        keygen(str(bad), tmp_path / "k", log=lambda *_: None)
    with pytest.raises(UnsupportedModel):
        load_model(str(bad))


def test_profile_round_trip(keys):
    key, public = keys
    prof = key.profile
    assert GemmaProfile.from_dict(prof.to_dict()) == prof
    assert prof.config_hash() == public["config_hash"]
    assert prof.head_dims == (16, 16, 16, 16, 16, 32) and prof.kv_heads == (2, 2, 2, 2, 2, 1)
