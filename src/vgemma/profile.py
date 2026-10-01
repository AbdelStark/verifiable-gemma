"""Gemma 4 text-decoder profile: support validation (fail closed), per-layer shapes, config hash.

The profile is derived from a transformers text config by keygen and by the server, and
reconstructed from its own canonical dict by the verifier, which never needs transformers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from vgemma.canon import canonical_json, embed_scale_bf16, json_hash

SUPPORTED_MODEL_TYPES = ("gemma4_text", "gemma4_unified_text")
SLIDING = "sliding_attention"
GLOBAL = "full_attention"
SUPPORTED_ROPE_KEYS = {"rope_type", "rope_theta", "partial_rotary_factor", "factor"}

# Matrix families: key name, module suffix under `layers.{i}.`, reason code.
FAMILIES: tuple[tuple[str, str, str], ...] = (
    ("wq", "self_attn.q_proj", "FREIVALDS_WQ"),
    ("wk", "self_attn.k_proj", "FREIVALDS_WK"),
    ("wv", "self_attn.v_proj", "FREIVALDS_WV"),
    ("wo", "self_attn.o_proj", "FREIVALDS_WO"),
    ("wgate", "mlp.gate_proj", "FREIVALDS_WGATE"),
    ("wup", "mlp.up_proj", "FREIVALDS_WUP"),
    ("wdown", "mlp.down_proj", "FREIVALDS_WDOWN"),
)
FAMILY_MODULE = {f: m for f, m, _ in FAMILIES}
FAMILY_CODE = {f: c for f, _, c in FAMILIES}

# Per-layer norm weights held in the key (v_norm has no weight).
LAYER_NORMS = (
    "input_layernorm",
    "post_attention_layernorm",
    "pre_feedforward_layernorm",
    "post_feedforward_layernorm",
    "self_attn.q_norm",
    "self_attn.k_norm",
)

# Captured tensors per position and layer, in leaf order (TECH_SPEC section 5).
#   r_in   input_layernorm input (residual stream into the layer)
#   x_attn input_layernorm output = q/k/v_proj input
#   q k v  q/k/v_proj outputs (v only on sliding layers)
#   q_n k_n v_n  q/k/v_norm outputs, pre-RoPE, [heads, head_dim]
#   a      o_proj input (attention output), o: o_proj output = post_attention_layernorm input
#   o_n    post_attention_layernorm output
#   r_mid  pre_feedforward_layernorm input (= r_in + o_n), x_ffn: its output = gate/up input
#   g u    gate/up outputs, h: down_proj input, d: down_proj output = post_ffn_layernorm input
#   d_n    post_feedforward_layernorm output; layer output = bf16(bf16(r_mid + d_n) * layer_scalar)
SLIDING_NAMES = (
    "r_in",
    "x_attn",
    "q",
    "k",
    "v",
    "q_n",
    "k_n",
    "v_n",
    "a",
    "o",
    "o_n",
    "r_mid",
    "x_ffn",
    "g",
    "u",
    "h",
    "d",
    "d_n",
)
GLOBAL_NAMES = tuple(n for n in SLIDING_NAMES if n != "v")
FINAL_NAMES = ("r_final", "h_final")  # final norm input and output


class UnsupportedModel(ValueError):
    """Raised for any configuration this profile cannot verify (fail closed)."""


@dataclass(frozen=True)
class GemmaProfile:
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    layer_types: tuple[str, ...]
    head_dims: tuple[int, ...]
    kv_heads: tuple[int, ...]
    sliding_window: int
    rope_parameters: dict[str, dict[str, Any]]
    rms_norm_eps: float
    final_logit_softcapping: float
    embed_scale_bf16: float
    attention_k_eq_v: bool
    hidden_activation: str

    # -- construction -------------------------------------------------------------------------

    @classmethod
    def from_hf_config(cls, cfg: Any) -> GemmaProfile:
        """Validate a transformers Gemma 4 text config and derive the profile (fail closed)."""
        if hasattr(cfg, "get_text_config"):
            cfg = cfg.get_text_config()
        model_type = getattr(cfg, "model_type", None)
        if model_type not in SUPPORTED_MODEL_TYPES:
            raise UnsupportedModel(
                f"model_type {model_type!r} is not a supported Gemma 4 text decoder "
                f"(expected one of {SUPPORTED_MODEL_TYPES})"
            )
        if getattr(cfg, "enable_moe_block", False):
            raise UnsupportedModel("Mixture-of-experts Gemma 4 (enable_moe_block) is not supported")
        if (getattr(cfg, "hidden_size_per_layer_input", 0) or 0) > 0:
            raise UnsupportedModel(
                "per-layer input embeddings (hidden_size_per_layer_input > 0, E2B/E4B) are not supported"
            )
        if (getattr(cfg, "num_kv_shared_layers", 0) or 0) > 0:
            raise UnsupportedModel("KV-shared layers (num_kv_shared_layers > 0) are not supported")
        if getattr(cfg, "use_double_wide_mlp", False):
            raise UnsupportedModel("use_double_wide_mlp is not supported")
        if getattr(cfg, "use_bidirectional_attention", None) == "all":
            raise UnsupportedModel("bidirectional text attention (use_bidirectional_attention='all') is not supported")
        if getattr(cfg, "attention_bias", False):
            raise UnsupportedModel("attention_bias is not supported")
        if getattr(cfg, "hidden_activation", None) != "gelu_pytorch_tanh":
            raise UnsupportedModel(f"hidden_activation {cfg.hidden_activation!r} is not gelu_pytorch_tanh")
        if not getattr(cfg, "tie_word_embeddings", False):
            raise UnsupportedModel("untied LM head is not supported (Gemma 4 ties lm_head to embed_tokens)")
        if not getattr(cfg, "attention_k_eq_v", False):
            raise UnsupportedModel(
                "attention_k_eq_v is false but the gemma4 profile expects shared K and V on global layers"
            )
        cap = getattr(cfg, "final_logit_softcapping", None)
        if cap is None or float(cap) <= 0:
            raise UnsupportedModel("final_logit_softcapping must be set (the profile replays the soft-cap)")

        layer_types = tuple(cfg.layer_types)
        unknown = set(layer_types) - {SLIDING, GLOBAL}
        if unknown:
            raise UnsupportedModel(f"unknown layer_types {sorted(unknown)}")
        if len(layer_types) != cfg.num_hidden_layers:
            raise UnsupportedModel("len(layer_types) != num_hidden_layers")
        if layer_types[-1] != GLOBAL:
            raise UnsupportedModel("the last layer must be full_attention")

        rope = {}
        for lt in sorted(set(layer_types)):
            params = dict(cfg.rope_parameters[lt])
            extra = set(params) - SUPPORTED_ROPE_KEYS
            if extra:
                raise UnsupportedModel(f"unsupported rope parameter keys for {lt}: {sorted(extra)}")
            if params.get("rope_type", "default") not in ("default", "proportional"):
                raise UnsupportedModel(f"unsupported rope_type {params.get('rope_type')!r} for {lt}")
            rope[lt] = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in params.items()}

        head_dims, kv_heads = [], []
        for i in range(cfg.num_hidden_layers):
            layer_cfg = cfg.per_layer_config[i]
            head_dims.append(int(layer_cfg.head_dim))
            kv_heads.append(int(layer_cfg.num_key_value_heads))
            if cfg.num_attention_heads % kv_heads[-1]:
                raise UnsupportedModel(f"layer {i}: num_attention_heads not divisible by KV heads")

        return cls(
            model_type=model_type,
            vocab_size=int(cfg.vocab_size),
            hidden_size=int(cfg.hidden_size),
            intermediate_size=int(cfg.intermediate_size),
            num_layers=int(cfg.num_hidden_layers),
            num_heads=int(cfg.num_attention_heads),
            layer_types=layer_types,
            head_dims=tuple(head_dims),
            kv_heads=tuple(kv_heads),
            sliding_window=int(cfg.sliding_window),
            rope_parameters=rope,
            rms_norm_eps=float(cfg.rms_norm_eps),
            final_logit_softcapping=float(cap),
            embed_scale_bf16=embed_scale_bf16(int(cfg.hidden_size)),
            attention_k_eq_v=True,
            hidden_activation=cfg.hidden_activation,
        )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GemmaProfile:
        d = dict(d)
        for k in ("layer_types", "head_dims", "kv_heads"):
            d[k] = tuple(d[k])
        return cls(**d)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("layer_types", "head_dims", "kv_heads"):
            d[k] = list(d[k])
        return d

    # -- derived quantities -------------------------------------------------------------------

    def is_global(self, layer: int) -> bool:
        return self.layer_types[layer] == GLOBAL

    def has_v_proj(self, layer: int) -> bool:
        return not self.is_global(layer)

    def families(self, layer: int) -> tuple[str, ...]:
        return tuple(f for f, _, _ in FAMILIES if f != "wv" or self.has_v_proj(layer))

    def family_shape(self, layer: int, family: str) -> tuple[int, int]:
        """(out_features, in_features) of the matrix."""
        hd, kv, h = self.head_dims[layer], self.kv_heads[layer], self.num_heads
        d, i = self.hidden_size, self.intermediate_size
        return {
            "wq": (h * hd, d),
            "wk": (kv * hd, d),
            "wv": (kv * hd, d),
            "wo": (d, h * hd),
            "wgate": (i, d),
            "wup": (i, d),
            "wdown": (d, i),
        }[family]

    def capture_names(self, layer: int) -> tuple[str, ...]:
        return GLOBAL_NAMES if self.is_global(layer) else SLIDING_NAMES

    def row_shape(self, layer: int, name: str) -> tuple[int, ...]:
        """Shape of one position's captured tensor; ``layer == num_layers`` is the final norm group."""
        d = self.hidden_size
        if layer == self.num_layers:
            return (d,)
        hd, kv, h, i = self.head_dims[layer], self.kv_heads[layer], self.num_heads, self.intermediate_size
        return {
            "r_in": (d,),
            "x_attn": (d,),
            "q": (h * hd,),
            "k": (kv * hd,),
            "v": (kv * hd,),
            "q_n": (h, hd),
            "k_n": (kv, hd),
            "v_n": (kv, hd),
            "a": (h * hd,),
            "o": (d,),
            "o_n": (d,),
            "r_mid": (d,),
            "x_ffn": (d,),
            "g": (i,),
            "u": (i,),
            "h": (i,),
            "d": (d,),
            "d_n": (d,),
        }[name]

    def group_names(self, layer: int) -> tuple[str, ...]:
        return FINAL_NAMES if layer == self.num_layers else self.capture_names(layer)

    def window(self, layer: int, pos: int) -> range:
        """Key positions attended by query ``pos`` (causal; sliding layers keep ``kv > q - W``)."""
        if self.is_global(layer):
            return range(0, pos + 1)
        return range(max(0, pos - self.sliding_window + 1), pos + 1)

    def config_dict(self) -> dict[str, Any]:
        d = self.to_dict()
        d.update(
            {
                "family": "gemma4",
                "attention_scaling": 1.0,
                "qk_norm": True,
                "v_norm": True,
                "rmsnorm": "x*(mean(x^2)+eps)^-0.5*w",
                "layer_scalar": "checkpoint",
            }
        )
        return d

    def config_hash(self) -> str:
        return json_hash("vg/config", self.config_dict()).hex()

    def layer_types_hash(self) -> str:
        return json_hash("vg/layer_types", list(self.layer_types)).hex()

    def rope_hash(self) -> str:
        per_type = {
            lt: {"params": self.rope_parameters[lt], "head_dim": self.head_dims[self.layer_types.index(lt)]}
            for lt in sorted(set(self.layer_types))
        }
        return json_hash("vg/rope", per_type).hex()

    def summary(self) -> str:
        n_global = sum(self.is_global(i) for i in range(self.num_layers))
        return canonical_json(
            {
                "layers": self.num_layers,
                "global_layers": n_global,
                "hidden": self.hidden_size,
                "heads": self.num_heads,
                "head_dim": sorted(set(self.head_dims)),
                "kv_heads": sorted(set(self.kv_heads)),
                "vocab": self.vocab_size,
            }
        )
