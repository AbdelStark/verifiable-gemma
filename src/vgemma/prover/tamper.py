"""Tamper modes for demonstrations and the adversarial suite (TECH_SPEC section 8).

Only reachable through an explicit ``--tamper`` flag; every mode logs a loud warning. The four
demo modes are ``weights``, ``identity``, ``sampling`` and ``softcap``; the rest exist so the
adversarial suite can exercise specific verifier checks.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from vgemma.model import LoadedModel
from vgemma.prover.sampler import SamplingPolicy

DEMO_MODES = ("weights", "identity", "sampling", "softcap")
TAMPER_MODES = (*DEMO_MODES, "identity-root", "rope-full", "no-vnorm")
EXPECTED_CODES = {
    "weights": "FREIVALDS_WDOWN",
    "identity": "FREIVALDS_WQ",
    "identity-root": "WEIGHTS_ROOT",
    "sampling": "DECODE_SAMPLING",
    "softcap": "DECODE_SOFTCAP",
    "rope-full": "ATTN_REPLAY",
    "no-vnorm": "BRIDGE_NORM_V",
}

BANNER = "!" * 78


@dataclass(frozen=True)
class Tamper:
    mode: str
    layer: int | None = None  # weights: layer whose down_proj gets the delta (default: middle layer)
    rel_norm: float = 0.03  # weights: ||delta||_F / ||W||_F
    rank: int = 8
    temperature_factor: float = 0.25  # sampling: actual temperature = declared * factor
    seed: int = 1234

    def __post_init__(self):
        if self.mode not in TAMPER_MODES:
            raise ValueError(f"unknown tamper mode {self.mode!r}; choose from {TAMPER_MODES}")

    def warn(self, log=print) -> None:
        log(BANNER)
        log(f"!!! TAMPER MODE '{self.mode}' ENABLED: this prover deliberately cheats (demo / tests only)")
        log(BANNER)

    # -- load-time model changes -----------------------------------------------------------------

    def apply_to_model(self, lm: LoadedModel, log=print) -> dict:
        import torch

        p = lm.profile
        if self.mode == "weights":
            layer = p.num_layers // 2 if self.layer is None else self.layer
            w = lm.layer_module(layer, "mlp.down_proj").weight
            g = torch.Generator().manual_seed(self.seed)
            a = torch.randn(w.shape[0], self.rank, generator=g, dtype=torch.float64)
            b = torch.randn(w.shape[1], self.rank, generator=g, dtype=torch.float64)
            delta = a @ b.T
            wf = w.detach().to(torch.float64).cpu()
            delta *= self.rel_norm * wf.norm() / delta.norm()
            with torch.no_grad():
                w.copy_((wf + delta).to(w.dtype).to(w.device))
            log(f"tamper weights: rank-{self.rank} delta, {self.rel_norm:.1%} of ||W||_F, layer {layer} mlp.down_proj")
            return {"layer": layer}
        if self.mode in ("identity", "identity-root"):
            a, b = self.swap_layers(lm)
            for suffix in (
                "input_layernorm",
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.q_norm",
                "self_attn.k_norm",
                "self_attn.o_proj",
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
                "post_feedforward_layernorm",
            ):
                wa, wb = lm.layer_module(a, suffix).weight, lm.layer_module(b, suffix).weight
                with torch.no_grad():
                    tmp = wa.detach().clone()
                    wa.copy_(wb)
                    wb.copy_(tmp)
            log(f"tamper identity: serving a different checkpoint (layers {a} and {b} swapped)")
            return {"swap": [a, b]}
        if self.mode == "rope-full":
            rot = lm.text_model.rotary_emb
            hd = p.head_dims[p.layer_types.index("full_attention")]
            theta = p.rope_parameters["full_attention"]["rope_theta"]
            inv = 1.0 / (theta ** (torch.arange(0, hd, 2, dtype=torch.float) / hd))
            rot.full_attention_inv_freq.copy_(inv.to(rot.full_attention_inv_freq.device))
            log("tamper rope-full: global layers use full RoPE instead of partial 0.25")
            return {}
        if self.mode == "no-vnorm":
            for i in range(p.num_layers):
                lm.layer_module(i, "self_attn.v_norm").forward = lambda x: x
            log("tamper no-vnorm: v_norm skipped on every layer")
            return {}
        return {}

    def swap_layers(self, lm: LoadedModel) -> tuple[int, int]:
        sliding = [i for i, t in enumerate(lm.profile.layer_types) if t == "sliding_attention"]
        return sliding[0], sliding[1]

    def renamed_tensor(self, name: str, prefix: str, lm: LoadedModel) -> str:
        """identity-root: the checkpoint name whose content the served model holds under ``name``."""
        a, b = self.swap_layers(lm)
        for x, y in ((a, b), (b, a)):
            head = f"{prefix}layers.{x}."
            if name.startswith(head) and not name.endswith("layer_scalar"):
                return f"{prefix}layers.{y}." + name[len(head) :]
        return name

    # -- run-time behaviour -----------------------------------------------------------------------

    def sampling_policy(self, declared: SamplingPolicy) -> SamplingPolicy:
        if self.mode != "sampling":
            return declared
        if declared.greedy:
            return replace(declared, greedy=False, temperature=1.0)
        return replace(declared, temperature=declared.temperature * self.temperature_factor)

    def postcap(self, precap: np.ndarray, honest: np.ndarray) -> np.ndarray:
        return precap.copy() if self.mode == "softcap" else honest
