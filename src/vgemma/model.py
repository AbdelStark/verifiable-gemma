"""Checkpoint resolution, model loading and module discovery by name pattern.

The text decoder is located by the pattern ``<prefix>layers.{i}.input_layernorm`` where
``<prefix>embed_tokens`` also exists, so the same code handles ``Gemma4ForCausalLM`` (tiny),
``Gemma4ForConditionalGeneration`` (31B) and ``Gemma4UnifiedForConditionalGeneration`` (12B).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from vgemma.profile import GemmaProfile, UnsupportedModel

LAYER_RE = re.compile(r"^(?P<prefix>(?:.*\.)?)layers\.(?P<idx>\d+)\.input_layernorm$")
# Module suffixes under `layers.{i}.` that the capture plan hooks (v_proj only on sliding layers).
LAYER_MODULES = (
    "input_layernorm",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.q_norm",
    "self_attn.k_norm",
    "self_attn.v_norm",
    "self_attn.o_proj",
    "post_attention_layernorm",
    "pre_feedforward_layernorm",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
    "post_feedforward_layernorm",
)


def resolve_model_dir(model: str) -> Path:
    """A local checkpoint directory, or a Hub id downloaded (once) into the HF cache."""
    p = Path(model)
    if p.is_dir():
        return p.resolve()
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(model, allow_patterns=["*.json", "*.safetensors", "*.jinja", "tokenizer*", "*.model"])
    )


def resolve_tokenizer_dir(model: str) -> Path:
    """Only the small public files a client needs to template its prompt (no weights)."""
    p = Path(model)
    if p.is_dir():
        return p.resolve()
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model, allow_patterns=["*.json", "*.jinja", "tokenizer*", "*.model"]))


def checkpoint_revision(model_dir: Path) -> str:
    # Hub snapshots live in .../snapshots/<commit sha>/
    if model_dir.parent.name == "snapshots":
        return model_dir.name
    return "local"


def checkpoint_files(model_dir: Path) -> list[Path]:
    files = sorted(Path(model_dir).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors files in {model_dir}")
    return files


def load_profile(model_dir: Path) -> tuple[GemmaProfile, object]:
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(str(model_dir))
    return GemmaProfile.from_hf_config(cfg), cfg


def eos_token_ids(model_dir: Path, cfg: object) -> list[int]:
    ids: set[int] = set()
    for c in (cfg, getattr(cfg, "get_text_config", lambda: cfg)()):
        e = getattr(c, "eos_token_id", None)
        ids.update([e] if isinstance(e, int) else (e or []))
    gen = Path(model_dir) / "generation_config.json"
    if gen.exists():
        e = json.loads(gen.read_text()).get("eos_token_id")
        ids.update([e] if isinstance(e, int) else (e or []))
    return sorted(ids)


# ---------------------------------------------------------------------------
# Streaming tensors from safetensors (keygen, weights root)
# ---------------------------------------------------------------------------


def to_numpy(t) -> np.ndarray:
    """A torch tensor as numpy; bfloat16 becomes uint16 bit patterns."""
    import torch

    t = t.detach().contiguous().cpu()
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16).numpy().view(np.uint16)
    return t.numpy()


def iter_checkpoint_tensors(model_dir: Path) -> Iterator[tuple[str, np.ndarray]]:
    """Every tensor of the checkpoint, sorted by name, as numpy (bf16 as uint16 bits)."""
    from safetensors import safe_open

    where: dict[str, Path] = {}
    for f in checkpoint_files(model_dir):
        with safe_open(str(f), framework="pt") as h:
            for k in h.keys():  # noqa: SIM118 (safe_open is not a mapping)
                if k in where:
                    raise ValueError(f"tensor {k} appears in two checkpoint files")
                where[k] = f
    handles: dict[Path, object] = {}
    try:
        for name in sorted(where):
            f = where[name]
            if f not in handles:
                handles[f] = safe_open(str(f), framework="pt").__enter__()
            yield name, to_numpy(handles[f].get_tensor(name))
    finally:
        for h in handles.values():
            h.__exit__(None, None, None)


def text_prefix(names: list[str] | set[str]) -> str:
    """Prefix of the text decoder in checkpoint tensor names (``model.`` or ``model.language_model.``)."""
    names = set(names)
    cands = sorted(
        n[: -len("embed_tokens.weight")]
        for n in names
        if n.endswith("embed_tokens.weight")
        and n[: -len("embed_tokens.weight")] + "layers.0.input_layernorm.weight" in names
    )
    if len(cands) != 1:
        raise UnsupportedModel(f"could not locate a unique text decoder in checkpoint (candidates {cands})")
    return cands[0]


# ---------------------------------------------------------------------------
# Loaded model and module map
# ---------------------------------------------------------------------------


@dataclass
class LoadedModel:
    model: object
    text_model: object
    lm_head: object
    final_norm: object
    profile: GemmaProfile
    hf_config: object
    model_dir: Path
    model_id: str
    revision: str
    device: str
    attn_implementation: str
    eos_token_ids: list[int]
    modules: dict[tuple[int, str], object] = field(default_factory=dict)

    def layer_module(self, layer: int, suffix: str):
        return self.modules[(layer, suffix)]


def discover(model, profile: GemmaProfile) -> tuple[str, dict[tuple[int, str], object]]:
    names = dict(model.named_modules())
    prefixes: dict[str, set[int]] = {}
    for n in names:
        m = LAYER_RE.match(n)
        if m:
            prefixes.setdefault(m["prefix"], set()).add(int(m["idx"]))
    text = [p for p in prefixes if (p + "embed_tokens") in names and (p + "norm") in names]
    if len(text) != 1:
        raise UnsupportedModel(f"could not locate a unique text decoder (candidates {sorted(prefixes)})")
    prefix = text[0]
    idx = prefixes[prefix]
    if idx != set(range(profile.num_layers)):
        raise UnsupportedModel(
            f"discovered {len(idx)} decoder layers, text_config.num_hidden_layers = {profile.num_layers}"
        )
    modules: dict[tuple[int, str], object] = {}
    for i in range(profile.num_layers):
        for suffix in LAYER_MODULES:
            full = f"{prefix}layers.{i}.{suffix}"
            if suffix == "self_attn.v_proj":
                present = full in names and names[full] is not None
                if present != profile.has_v_proj(i):
                    raise UnsupportedModel(
                        f"layer {i} ({profile.layer_types[i]}): v_proj present={present}, "
                        f"profile expects {profile.has_v_proj(i)}"
                    )
                if not present:
                    continue
            if full not in names:
                raise UnsupportedModel(f"missing module {full}")
            modules[(i, suffix)] = names[full]
    return prefix, modules


def load_model(model: str, device: str = "cpu", attn_implementation: str = "sdpa") -> LoadedModel:
    import torch
    from transformers import AutoModelForCausalLM

    model_dir = resolve_model_dir(model)
    profile, cfg = load_profile(model_dir)
    kw = {"device_map": device} if device.startswith("cuda") else {}
    hf = AutoModelForCausalLM.from_pretrained(
        str(model_dir), dtype=torch.bfloat16, attn_implementation=attn_implementation, **kw
    )
    if not kw:
        hf.to(device)
    hf.eval()
    prefix, modules = discover(hf, profile)
    named = dict(hf.named_modules())
    for i in range(profile.num_layers):
        layer = named[f"{prefix}layers.{i}"]
        if getattr(layer, "layer_scalar", None) is None or layer.layer_scalar.dtype != torch.bfloat16:
            raise UnsupportedModel(f"layer {i}: expected a bfloat16 layer_scalar buffer (residual replay assumes it)")
    text_model = named[prefix.rstrip(".")]
    lm_head = named.get("lm_head")
    if lm_head is None:
        raise UnsupportedModel("no lm_head module found")
    return LoadedModel(
        model=hf,
        text_model=text_model,
        lm_head=lm_head,
        final_norm=named[prefix + "norm"],
        profile=profile,
        hf_config=cfg,
        model_dir=model_dir,
        model_id=model if not Path(model).is_dir() else model_dir.name,
        revision=checkpoint_revision(model_dir),
        device=device,
        attn_implementation=attn_implementation,
        eos_token_ids=eos_token_ids(model_dir, cfg),
        modules=modules,
    )
