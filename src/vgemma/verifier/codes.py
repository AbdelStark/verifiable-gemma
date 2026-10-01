"""Reason codes (TECH_SPEC section 10). Add a code to the spec before using it here."""

from __future__ import annotations

CODES: dict[str, str] = {
    "RECEIPT_SCHEMA": "receipt is malformed or has an unsupported version",
    "RECEIPT_SIGNATURE": "prover signature over the receipt hash does not verify (or prover id not the pinned one)",
    "OPENING_SCHEMA": "opening is malformed, for another request, or does not answer the challenge",
    "WEIGHTS_ROOT": "receipt weights root differs from the public params",
    "CONFIG_HASH": "receipt config hash differs from the public params",
    "MANIFEST_UNSUPPORTED": "deployment manifest declares an unsupported mode",
    "MANIFEST_MISMATCH": "manifest field differs from the value derived from the checkpoint",
    "SEED_COMMITMENT": "revealed seed does not open the seed commitment",
    "PROMPT_BINDING": "opened prompt tokens do not match the prompt hash or token count",
    "IO_CHAIN": "token transcript does not reproduce the IO chain head, count or stop rule",
    "MERKLE_POSITION": "opened data does not reproduce the committed position leaf and trace root",
    "EMBEDDING": "embedding row proof or scaled-embedding replay fails",
    "FREIVALDS_WQ": "Freivalds check on the query projection fails",
    "FREIVALDS_WK": "Freivalds check on the key projection fails",
    "FREIVALDS_WV": "Freivalds check on the value projection fails",
    "FREIVALDS_WO": "Freivalds check on the output projection fails",
    "FREIVALDS_WGATE": "Freivalds check on the gate projection fails",
    "FREIVALDS_WUP": "Freivalds check on the up projection fails",
    "FREIVALDS_WDOWN": "Freivalds check on the down projection fails",
    "BRIDGE_NORM_INPUT": "input_layernorm replay fails",
    "BRIDGE_NORM_POST_ATTN": "post_attention_layernorm replay fails",
    "BRIDGE_NORM_PRE_FFN": "pre_feedforward_layernorm replay fails",
    "BRIDGE_NORM_POST_FFN": "post_feedforward_layernorm replay fails",
    "BRIDGE_NORM_Q": "q_norm replay fails",
    "BRIDGE_NORM_K": "k_norm replay fails",
    "BRIDGE_NORM_V": "v_norm (no weight) replay fails on a sliding layer",
    "BRIDGE_NORM_FINAL": "final norm replay fails",
    "BRIDGE_GELU": "GELU-tanh gate replay fails",
    "BRIDGE_RESIDUAL": "residual add or layer_scalar replay fails (exact)",
    "WIRING": "layer type, head dims, KV heads, shared K/V, RoPE or QK-norm wiring mismatch",
    "KV_PROVENANCE": "an attended K/V row is missing or not bound to the trace root",
    "ATTN_REPLAY": "single-query attention replay deviates beyond the audit tolerance",
    "KV_SHARED": "global-layer V is not v_norm(K before k_norm)",
    "LMHEAD_BINDING": "captured logits are not bound to the final hidden state by the LM head",
    "DECODE_SOFTCAP": "opened post-cap logits are not the canonical soft-cap of the pre-cap logits",
    "DECODE_SAMPLING": "shared sampler on the opened logits does not reproduce the committed token",
}

SHELL = {
    "wq": "FREIVALDS_WQ",
    "wk": "FREIVALDS_WK",
    "wv": "FREIVALDS_WV",
    "wo": "FREIVALDS_WO",
    "wgate": "FREIVALDS_WGATE",
    "wup": "FREIVALDS_WUP",
    "wdown": "FREIVALDS_WDOWN",
}

AUDITED = frozenset({"WIRING", "KV_PROVENANCE", "ATTN_REPLAY"})
TOLERANCE = frozenset(
    {*SHELL.values(), "LMHEAD_BINDING", "KV_SHARED", "BRIDGE_GELU"} | {c for c in CODES if c.startswith("BRIDGE_NORM_")}
)


def kind_of(code: str) -> str:
    """exact (bit-exact or a binding), tolerance (bound printed), or audited (not a verification)."""
    return "audited" if code in AUDITED else "tolerance" if code in TOLERANCE else "exact"
