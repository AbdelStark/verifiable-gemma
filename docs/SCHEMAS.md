# Schemas (version 1)

All hashes are SHA-256 with domain separation, `H(tag, parts...) = SHA-256(len(tag) || tag || parts...)`,
and are written as lowercase hex. Hashed JSON is canonical: `json.dumps(obj, sort_keys=True, separators=(",", ":"))`.
Integers inside hashes are little-endian `u32` (the sampled token is `i32`, with -1 for none).
bfloat16 tensors travel as `uint16` bit patterns.

## Canonical tensor bytes

`dtype_tag (u8) || ndim (u32) || shape (u32 each) || raw little-endian bytes`, with tags
`bf16=1 f32=2 f64=3 i32=4 i64=5 u8=6 i8=7 f16=8 bool=9`. `H_t(x) = H("vg/tensor", bytes(x))`.

## Commitments

| Name | Definition |
|---|---|
| weight leaf | `H("vg/weight", u32(len(name)), name, bytes(tensor))` for every tensor of the safetensors files |
| weights root | Merkle root of the weight leaves sorted by name |
| embedding root | Merkle root of `H_t(E[i])` for every vocabulary row |
| layer leaf | `H("vg/layer", u32(layer), u32(pos), H_t(t) for t in profile order)`; layer `num_layers` is the final norm group `r_final, h_final` |
| witness hash | `H("vg/witness", canonical_json({"step", "u", "postcap": H_t(postcap) hex}))` |
| position body | `H("vg/pos_body", layer leaves 0..num_layers, H_t(logits_precap) or 0^32, i32(sampled token or -1), witness hash or 0^32)` |
| position leaf | `H("vg/pos", u32(pos), u32(input_token), body)` |
| trace root | Merkle root of the position leaves, positions `0 .. n_prompt + n_gen - 2` |
| Merkle node | `H("vg/node", left, right)`; padding to a power of two with `H("vg/leaf", "empty")` |
| prompt hash | `H("vg/prompt", u32(n), u32 token ids...)` over the chat-templated prompt |
| IO chain | `c_0 = H("vg/io", prompt_hash)`, `c_t = H("vg/io", c_{t-1}, u32(token_t), H_t(logits_precap_t))` |
| seed | with a client nonce `H("vg/seed_client", nonce)`, otherwise `H("vg/seed", prover_secret, request_id)`; commitment `H("vg/seed_commit", seed, request_id)` |
| uniform | `u_t = (le_u64(H("vg/u", seed, u32(t))[:8]) >> 11) / 2^53` |
| config hash | `H("vg/config", canonical_json(profile.config_dict()))` |
| receipt hash | `H("vg/receipt", canonical_json(receipt without prover.signature))`; Ed25519 signs it |

Per-layer capture order: `r_in x_attn q k v q_n k_n v_n a o o_n r_mid x_ffn g u h d d_n` (no `v` on
global layers). Row shapes: hidden-size vectors, `q`/`a` `[H·d]`, `k`/`v` `[KV·d]`, `q_n` `[H, d]`,
`k_n`/`v_n` `[KV, d]`, `g`/`u`/`h` `[intermediate]`, with `d` and `KV` per layer type.

## Receipt

```json
{
  "version": 1,
  "request_id": "r_9951b255b0362739",
  "model": {"id": "google/gemma-4-12B-it", "revision": "<snapshot sha>", "weights_root": "<hex>", "config_hash": "<hex>"},
  "manifest": {
    "dtype": "bfloat16", "attn_implementation": "sdpa",
    "temperature": 1.0, "top_p": 0.95, "top_k": 64, "greedy": false,
    "final_logit_softcapping": 30.0, "embed_scale_bf16": 62.0, "rms_norm_eps": 1e-06,
    "sliding_window": 1024, "layer_types_hash": "<hex>", "attention_k_eq_v": true,
    "rope_hash": "<hex>", "qk_norm": true, "thinking": false,
    "chat_template_hash": "<hex>", "tokenizer_hash": "<hex>",
    "speculative": "none", "prefix_caching": false,
    "max_new_tokens": 128, "eos_token_ids": [1, 50, 106],
    "sampler": "vg-sampler-1", "softcap_impl": "vg-canon-softcap-1", "lm_head": "f32"
  },
  "prompt_hash": "<hex>", "seed_commitment": "<hex>", "client_nonce": "<hex>" | null,
  "n_prompt": 57, "n_gen": 24,
  "trace_root": "<hex>", "io_chain_head": "<hex>",
  "prover": {"id": "ed25519:<public key hex>", "signature": "ed25519:<signature hex>"}
}
```

## Challenge (`POST /audit`)

```json
{"request_id": "r_...", "tier": "routine:3",
 "audits": [{"pos": 56, "layers": [0, 3, 5], "attention": true},
            {"pos": 57, "layers": [4], "attention": false}, ...]}
```

Every audit:

- verifies the embedding row of its input token;
- audits each of its layers in full (all tensors, Freivalds, bridges, both residual adds);
- with `attention`, runs the attention audit (K/V provenance and replay);
- if the position samples a token, runs the decode checks (final norm, LM-head binding, soft-cap,
  sampled token).

Position `p` samples token `p + 1`, so the decode positions are `n_prompt - 1 .. n_prompt + n_gen - 2`.

The auditor draws the challenge with its own randomness after it holds the receipt. By default
that is 3 random generated positions with full audits (an independent routine layer subset each,
plus attention), and a decode audit with one random layer for every other generated token. A
residual stream forged at one layer boundary at every position escapes with
`prod_a (1 - |layers_a| / L)`, which the verdict reports. The verifier is given the auditor's
challenge and rejects an opening whose echo differs, or a challenge that audits no layer.

## Client request (`vg chat` writes `request.json`)

```json
{"messages": [{"role": "user", "content": "..."}],
 "params": {"max_new_tokens": 128, "thinking": false, "greedy": false, "temperature": 1.0,
            "top_k": 64, "top_p": 0.95, "nonce": "<32-byte hex>"},
 "prover_id": "ed25519:<hex>", "prompt_tokens": [2, 105, ...],
 "tokenizer_hash": "<hex>", "chat_template_hash": "<hex>", "attn_implementation": "sdpa"}
```

The client templates the prompt itself with the checkpoint's public tokenizer (`vg chat --model`).
`vg verify --request` turns the request into `expected`:

- the policy fields, tokenizer and template hashes and the attention implementation bind
  `MANIFEST_MISMATCH`;
- the nonce binds `SEED_COMMITMENT`;
- the prompt tokens bind `PROMPT_BINDING`; a request without a prompt binding is rejected.

It also pins the prover id.

## Opening

Binary: `b"VGOPEN01" || u64 len(index) || index JSON || safetensors blob`.

Index:

```json
{
  "version": 1, "request_id": "r_...",
  "challenge": {"positions": [...], "layers": [...], "attention": true, "open_prompt": true, "tier": "..."},
  "seed": "<hex>",
  "prompt_tokens": [2, 3, ...],
  "io_transcript": [[token, "<H_t(logits_precap) hex>"], ...],
  "positions": {
    "<j>": {
      "input_token": 115, "out_token": 207,
      "logits": "open" | "<hex>" | null,
      "witness": {"step": 0, "u": 0.4127, "postcap": "<hex>"} | "<hex>" | null,
      "layers": {"<l>": {"leaf": "<hex>"} | {"tensors": {"<name>": "open" | "<hex>"}}},
      "proof": ["<hex>", ...],
      "embedding": {"proof": ["<hex>", ...]}
    }
  }
}
```

Tensors are keyed `p{j}/l{l}/{name}` (bf16 as U16), `p{j}/logits_precap` (F32, the f32 LM-head
output) and `p{j}/embed_row` (U16). Post-cap logits are not sent: the verifier recomputes them with
the canonical soft-cap and checks their hash against the sampler witness.

An opening contains an entry for every position `0 .. n_prompt + n_gen - 2`:

- for every audit: layer 0 `r_in`, the embedding row with its proof, every tensor of each audited
  layer, the next layer's `r_in` (or `r_final`), and at decode positions the final group and the
  logits;
- with `attention`, the `k_n`/`v_n` rows of every position in that layer's window (sliding:
  `kv > q - W`; global: all of `0..q`);
- every other position as a token-only entry `{"input_token", "body", "proof"}`. These bind every
  input token, prompt included, to the trace root, and the verifier checks each against the
  revealed prompt or the transcript.

A layer group is exactly `{"leaf": ...}` or exactly `{"tensors": ...}`, and every opened tensor must
be finite. `prompt_tokens` is always present.

## Verdict

```json
{
  "version": 1, "result": "PASS" | "FAIL", "reason": "FREIVALDS_WDOWN" | null, "message": "...",
  "layer": 3, "position": 85, "detail": {"deviation": 0.0178, "tolerance": 0.0073},
  "coverage": {"bindings": "verified", "embedding": "verified (exact)",
               "shell": "verified, tolerance-bounded (7/7 families, 6/6 layers, k=16)",
               "bridge": "...", "attention": "audited (...)", "decode": "..."},
  "checks": {"FREIVALDS_WQ": {"kind": "tolerance", "n": 18, "max_ratio": 0.185,
                              "worst": {"deviation": 0.0061, "tolerance": 0.0331, "layer": 2, "position": 56, "unit": ""}}},
  "payload_bytes": 397118, "verify_ms": 23.8, "request_id": "r_...",
  "audits": [{"pos": 56, "layers": [0, 3, 5], "attention": true}, ...], "positions": [56, 57, ...],
  "layers": [0, 1, 2, 3, 4, 5],
  "spot_check": {"audits": 24, "layer_audits": 30, "attention_audits": 3, "decode_checked": 24, "n_gen": 24,
                 "forged_boundary_escape": 0.0027, "fake_attention_escape": 0.125},
  "pinned_prover": true, "client_expectations": ["client_nonce", "greedy", "max_new_tokens", ...],
  "tolerances": {"freivalds": "...", "norms": "...", "gelu": "...", "kv_shared": "...", "attention": "..."}
}
```

`kind` is `exact` (bit-exact comparison or a binding), `tolerance` (deviation and bound printed) or
`audited` (wiring, K/V provenance, attention replay: not described as verification).

## Key and public params

`public.json` (given to the provider and to anyone): `version, model_id, revision, weights_root,
embedding_root, config_hash, n_weight_tensors, text_prefix, eos_token_ids, tokenizer_hash,
chat_template_hash, profile, keygen {seconds, key_bytes, freivalds_k}`.

`key.npz` (secret, auditor only): `r.{l}.{family}` int8 `[k, out]`, `v.{l}.{family}` float64 `[k, in]`
for the families of each layer type (no `wv` on global layers), `r.lm` int8 `[k, vocab]`, `v.lm` float64
`[k, hidden]`, norm weights `w.{l}.{norm}` and `w.final` (float32, exact bf16 values), layer scalars
`s.{l}`, and `meta` (the public fields plus `freivalds_k` and a key id).
