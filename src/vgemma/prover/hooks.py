"""Forward hooks that capture the tensors of the capture plan (TECH_SPEC section 5).

Captured tensors stay on the device in the model dtype; they are concatenated and moved to the CPU
in bulk once per request (`collect`), never per hook call.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from vgemma.model import LoadedModel, to_numpy

# module suffix -> [(input|output, capture name)]
CAPTURE_PLAN: dict[str, list[tuple[str, str]]] = {
    "input_layernorm": [("in", "r_in"), ("out", "x_attn")],
    "self_attn.q_proj": [("out", "q")],
    "self_attn.k_proj": [("out", "k")],
    "self_attn.v_proj": [("out", "v")],
    "self_attn.q_norm": [("out", "q_n")],
    "self_attn.k_norm": [("out", "k_n")],
    "self_attn.v_norm": [("out", "v_n")],
    "self_attn.o_proj": [("in", "a"), ("out", "o")],
    "post_attention_layernorm": [("out", "o_n")],
    "pre_feedforward_layernorm": [("in", "r_mid"), ("out", "x_ffn")],
    "mlp.gate_proj": [("out", "g")],
    "mlp.up_proj": [("out", "u")],
    "mlp.down_proj": [("in", "h"), ("out", "d")],
    "post_feedforward_layernorm": [("out", "d_n")],
}
FINAL_PLAN = [("in", "r_final"), ("out", "h_final")]


class CaptureHooks:
    def __init__(self, lm: LoadedModel):
        self.lm = lm
        self.profile = lm.profile
        self.enabled = False
        self.chunks: dict[tuple[int, str], list] = defaultdict(list)
        self.handles = []

    def install(self) -> None:
        for (layer, suffix), module in self.lm.modules.items():
            self.handles.append(module.register_forward_hook(self._hook(layer, CAPTURE_PLAN[suffix])))
        final = self.profile.num_layers
        self.handles.append(self.lm.final_norm.register_forward_hook(self._hook(final, FINAL_PLAN)))

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def _hook(self, layer: int, specs: list[tuple[str, str]]):
        def hook(module, args, output):
            if not self.enabled:
                return
            for io, name in specs:
                t = args[0] if io == "in" else output
                if t.shape[0] != 1:
                    raise RuntimeError("capture supports batch size 1 only")
                self.chunks[(layer, name)].append(t.detach()[0].clone())

        return hook

    def reset(self) -> None:
        self.chunks.clear()

    def collect(self, n_positions: int) -> dict[int, dict[str, np.ndarray]]:
        """Concatenate per-forward chunks into ``[n_positions, *row_shape]`` bf16 bit arrays."""
        import torch

        out: dict[int, dict[str, np.ndarray]] = {}
        p = self.profile
        for layer in range(p.num_layers + 1):
            group = {}
            for name in p.group_names(layer):
                chunks = self.chunks.get((layer, name))
                if not chunks:
                    raise RuntimeError(f"capture missing for layer {layer} tensor {name}")
                t = torch.cat(chunks, dim=0)
                if t.dtype != torch.bfloat16:
                    raise RuntimeError(f"capture {layer}/{name} has dtype {t.dtype}, expected bfloat16")
                arr = to_numpy(t)
                if arr.shape[0] != n_positions:
                    raise RuntimeError(f"capture {layer}/{name}: {arr.shape[0]} positions, expected {n_positions}")
                group[name] = arr.reshape(n_positions, *p.row_shape(layer, name))
            out[layer] = group
        return out
