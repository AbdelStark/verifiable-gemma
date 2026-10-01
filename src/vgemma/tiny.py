"""Tiny random Gemma 4 text checkpoint for CPU tests (TECH_SPEC section 12).

The configuration copies the Gemma 4 12B text config and shrinks the dimensions, keeping every
architectural feature: the 5:1 sliding/global pattern, a wider global head dim, shared K and V on
global layers, proportional partial RoPE, QK-norm and V-norm, the logit soft-cap, tied embeddings,
the bf16-rounded embedding scale, and non-trivial per-layer ``layer_scalar`` values (the published
checkpoints carry values far from 1).
"""

from __future__ import annotations

from pathlib import Path

from vgemma.tokenizer import TinyByteTokenizer

# Gemma 4 12B-it text_config fields kept as is (google/gemma-4-12B-it config.json).
GEMMA4_12B_TEXT_FIXED = {
    "attention_bias": False,
    "attention_dropout": 0.0,
    "attention_k_eq_v": True,
    "enable_moe_block": False,
    "final_logit_softcapping": 30.0,
    "hidden_activation": "gelu_pytorch_tanh",
    "hidden_size_per_layer_input": 0,
    "num_kv_shared_layers": 0,
    "rms_norm_eps": 1e-06,
    "rope_parameters": {
        "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
        "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
    },
    "tie_word_embeddings": True,
    "use_double_wide_mlp": False,
    "pad_token_id": 0,
    "bos_token_id": 2,
    "eos_token_id": 1,
}

TINY_DIMS = {
    "hidden_size": 64,
    "num_hidden_layers": 6,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "global_head_dim": 32,
    "num_global_key_value_heads": 1,
    "intermediate_size": 128,
    "vocab_size": 512,
    "sliding_window": 8,
    "max_position_embeddings": 4096,
}


def tiny_config(**overrides):
    from transformers import Gemma4TextConfig

    kw = {**GEMMA4_12B_TEXT_FIXED, **TINY_DIMS, **overrides}
    n = kw["num_hidden_layers"]
    kw.setdefault(
        "layer_types", ["full_attention" if (i + 1) % 6 == 0 or i == n - 1 else "sliding_attention" for i in range(n)]
    )
    return Gemma4TextConfig(**kw)


def build_tiny(out: Path, seed: int = 0, **overrides) -> Path:
    """Write a randomly initialised tiny checkpoint (safetensors, config, tokenizer) to ``out``."""
    import torch
    from transformers import Gemma4ForCausalLM, GenerationConfig

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = tiny_config(**overrides)
    torch.manual_seed(seed)
    model = Gemma4ForCausalLM(cfg)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("embed_tokens.weight"):
                p.copy_(torch.randn(p.shape, generator=g) * 0.5)
            elif "norm" in name:
                p.copy_(0.5 + torch.rand(p.shape, generator=g))
            elif p.ndim == 2:
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)
        for layer in model.model.layers:
            layer.layer_scalar.copy_(0.3 + 0.7 * torch.rand(layer.layer_scalar.shape, generator=g))
    model = model.to(torch.bfloat16)
    model.save_pretrained(str(out))
    tok = TinyByteTokenizer(vocab_size=cfg.vocab_size)
    tok.save(out)
    GenerationConfig(
        bos_token_id=2,
        eos_token_id=tok.eos_token_ids,
        pad_token_id=0,
        do_sample=True,
        temperature=1.0,
        top_k=64,
        top_p=0.95,
    ).save_pretrained(str(out))
    return out
