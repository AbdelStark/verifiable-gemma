# verifiable-gemma: Technical Specification

Status: draft v0.1. Companion to PRD.md and AGENTS.md.

## 1. Background: CommitLLM

CommitLLM (github.com/lambdaclass/CommitLLM, MIT) is a commit-and-audit protocol for open-weight LLM inference:

1. Setup. The verifier builds a key from the public checkpoint: a Merkle root over weights, secret Freivalds vectors per weight-matrix family (Wq, Wk, Wv, Wo, Wgate, Wup, Wdown, LM head), and the configuration needed for replay.
2. Commit. The provider runs the model normally, captures retained state, and returns a receipt binding the execution trace, KV state, deployment manifest, prompt, sampling randomness and token count.
3. Audit. When challenged, the provider opens requested token positions and layers.
4. Verify. The verifier checks embedding Merkle proofs, shell matmuls via Freivalds, bridge tensors by canonical recomputation, KV transcript proofs, decode from captured logits plus LM-head binding, and all bindings.

Its implementation is Rust (verilm-core, verilm-keygen, verilm-verify, verilm-prover) with a Python sidecar around vLLM, validated on Llama-3.1-8B W8A8 and Qwen2.5-7B W8A8 (INT8 weights and activations, i32 accumulators, Freivalds over the field p = 2^32 - 5). Its honest claim: exact decode plus audited attention inputs and wiring; everything else is verified with exact or information-theoretically sound checks; attention outputs at arbitrary positions are not verified on stock fused kernels. Measured: 12 to 14 percent tracing overhead, about 0.5 MiB retained per token (128k vocabulary), about 1.3 ms verifier time per challenged token on Llama 70B.

Gemma is listed in CommitLLM's roadmap as the third dense family, not started. verifiable-gemma implements the protocol for Gemma 4 in two tiers: Tier 1, a Python reference implementation on `transformers` in bf16 with tolerance-bounded linear checks and exact decode; Tier 2, the exact INT8 path inside CommitLLM.

## 2. Gemma 4 facts that drive the design

Source: `config.json` and `modeling_gemma4.py` for `google/gemma-4-31B-it` and `google/gemma-4-12B-it` (both Apache 2.0, ungated). Gemma 4 also ships 26B-A4B (MoE, 128 experts) and E2B/E4B (per-layer embeddings, 18 KV-shared layers); both are out of scope and must fail closed.

| Feature | 31B-it | 12B-it | Verification consequence |
|---|---|---|---|
| Model class | `Gemma4ForConditionalGeneration` (text decoder plus vision tower, `model_type gemma4`) | `Gemma4UnifiedForConditionalGeneration` (`model_type gemma4_unified`, text config `gemma4_unified_text`, same decoder) | Discover the text decoder by module-name pattern, not class |
| Layers, hidden, heads | 60, 5376, 32 | 48, 3840, 16 | Per-layer-type shapes in the key |
| Layer types | `sliding_attention` x5 then `full_attention`, repeating; final layer global | same | Pattern hashed in config; drives window and head_dim per layer |
| Sliding window | 1024 | 1024 | KV provenance scope per layer |
| Sliding layers | head_dim 256, 16 KV heads (31B) / 8 (12B), RoPE theta 1e4 default | | |
| Global layers | head_dim 512 (`global_head_dim`), 4 KV heads (31B) / 1 (12B), `attention_k_eq_v = true` (no v_proj; V = v_norm(K before k_norm and RoPE)), RoPE theta 1e6, `rope_type = proportional`, `partial_rotary_factor = 0.25` | | Freivalds family set differs per layer type; one exact extra check: V derives from K |
| Attention scaling | 1.0 (QK-norm replaces 1/sqrt(d)) | | Score replay uses no scale |
| QK-norm | `q_norm`, `k_norm` RMSNorm with weight per head_dim; `v_norm` RMSNorm without weight | | Hookable modules; weights in key |
| Decoder layer | `input_layernorm` -> attn -> `post_attention_layernorm` -> residual add -> `pre_feedforward_layernorm` -> MLP -> `post_feedforward_layernorm` -> residual add; then `hidden_states *= layer_scalar` (a bf16 checkpoint buffer, not ones: 12B layers 0/1/5/11/23/47 hold 0.053, 0.166, 0.355, 0.0045, 0.758, 0.048) | | Four norms per layer in the bridge; two apply to sub-block outputs; the residual replay multiplies by `layer_scalar` from the key |
| RMSNorm | `x * (mean(x^2) + eps)^-0.5 * w` in float32 then cast; eps 1e-6; some instances without `w` | | Canonical f64 replay; the Llama `(1 + w)` form does not apply here |
| MLP | `down_proj(gelu_tanh(gate_proj(x)) * up_proj(x))`, `hidden_activation = gelu_pytorch_tanh` | | GELU-tanh bridge instead of SiLU |
| Embedding | `embed_tokens(ids) * embed_scale`, `embed_scale = sqrt(hidden_size)` cast to the weight dtype (bf16: 73.5 for 5376, 62.0 for 3840) | | Replay the bf16-rounded scale, not the float value |
| LM head | tied to `embed_tokens`; `final_logit_softcapping = 30.0`: `logits = 30 * tanh(logits / 30)` | | Capture pre-cap logits for the binding; replay the cap for decode |
| Vocabulary | 262,144 | 262,144 | Logits are 1 MiB per token in f32 |
| Recommended sampling | temperature 1.0, top_p 0.95, top_k 64 | | Default manifest |
| Thinking | `<|think|>` control token at the start of the system prompt | | Bound through the prompt hash and a manifest flag; off by default |

Shapes, 31B, per layer: sliding Wq 8192x5376, Wk 4096x5376, Wv 4096x5376, Wo 5376x8192; global Wq 16384x5376, Wk 2048x5376, no Wv, Wo 5376x16384; Wgate and Wup 21504x5376, Wdown 5376x21504; tied embedding and LM head 262144x5376. Totals: 29.3B parameters in projections, 1.41B in the tied embedding.

Shapes, 12B, per layer: sliding Wq 4096x3840, Wk 2048x3840, Wv 2048x3840, Wo 3840x4096; global Wq 8192x3840, Wk 512x3840, no Wv, Wo 3840x8192; Wgate and Wup 15360x3840, Wdown 3840x15360; embedding 262144x3840.

Memory, bf16: 31B about 62 GB plus 1 GB vision tower (needs an 80 GB GPU); 12B about 24 GB (fits 40 GB and 48 GB GPUs).

## 3. Model selection

- Development and tests: tiny random Gemma 4 text config on CPU (section 12).
- Real-model default: `google/gemma-4-12B-it` (fits common single GPUs, same decoder design).
- Flagship: `google/gemma-4-31B-it` on an 80 GB GPU.
- Load the published checkpoint as is (vision tower included, unused); feed text only. A text-only `Gemma4ForCausalLM` export is an optional optimisation, not a requirement.
- Tier 2 artefact: a derived W8A8 checkpoint from the same base (section 15).

## 4. System architecture

```
OFFLINE
  public checkpoint ──vg keygen──> verifier key (secret, small)  +  public params (roots, config hash)

PROVIDER                                           AUDITOR
  vg serve: transformers + capture hooks            holds key + public params
    /chat  -> text + receipt                         vg audit  -> challenge {positions, layers}
    retains state per request (disk, TTL)           <- opening (tensors + proofs for challenged parts)
    /audit -> opening                                vg verify -> verdict JSON
    tamper modes for demonstration
```

Components:

| Component | Responsibility |
|---|---|
| `vgemma/profile.py` | Reads HF config, validates support (fail closed), derives per-layer shapes, layer types, RoPE parameters, soft-cap, embed scale, norm eps; computes the canonical config hash |
| `vgemma/model.py` | Loads the checkpoint, locates the text decoder, builds the module map by name pattern, exposes tiny mode |
| `vgemma/canon.py` | Canonical tensor serialisation and hashing (domain-separated), f64 reference functions: `rmsnorm_gemma`, `gelu_tanh`, `softcap`, `embed_scale`, RoPE tables |
| `vgemma/merkle.py` | Merkle tree over byte leaves, proofs, verification |
| `vgemma/keygen.py` | Weights root, embedding root, Freivalds vectors, norm weights, key and public-params files |
| `vgemma/prover/hooks.py` | Forward hooks that capture inputs and outputs of the hooked modules per position |
| `vgemma/prover/sampler.py` | CPU sampler shared with the verifier |
| `vgemma/prover/engine.py` | Own decode loop with KV cache, receipt construction, retained-state store, tamper modes |
| `vgemma/prover/server.py` | HTTP API |
| `vgemma/verifier/*.py` | Freivalds, bridge, attention, decode, bindings, orchestration, verdict |
| `vgemma/cli.py` | `vg keygen | serve | chat | audit | verify | demo | bench | tiny` (Typer; invoked as `uv run vg ...`) |
| `infra/modal_app.py` | Modal app: image from `uv.lock`, volumes, `download`, `tests_tiny`, `keygen`, `serve`, `demo`, `bench` functions |
| `infra/vast/` | vast.ai launcher: `setup.sh` and instance notes; same commands over SSH |

## 5. Capture plan (what the hooks record)

Hooks are registered on modules found by regex under the text decoder (`layers.{i}.<name>`), for every layer `i`:

| Module | Captured | Role |
|---|---|---|
| `input_layernorm` | input (residual r_in), output (x_attn) | bridge, residual chain |
| `self_attn.q_proj` | output (q) | Freivalds Wq; input is x_attn |
| `self_attn.k_proj` | output (k_pre) | Freivalds Wk; source of V on global layers |
| `self_attn.v_proj` (sliding only) | output (v_pre) | Freivalds Wv |
| `self_attn.q_norm`, `self_attn.k_norm`, `self_attn.v_norm` | output (q_n, k_n, v_n), pre-RoPE | attention replay, provenance |
| `self_attn.o_proj` | input (a), output (o) | Freivalds Wo; attention replay target |
| `post_attention_layernorm` | output | bridge |
| `pre_feedforward_layernorm` | input (r_mid), output (x_ffn) | residual chain; bridge; input to gate and up |
| `mlp.gate_proj`, `mlp.up_proj` | outputs (g, u) | Freivalds |
| `mlp.down_proj` | input (h), output (d) | GELU bridge; Freivalds Wdown |
| `post_feedforward_layernorm` | output | bridge; residual out = r_mid + output |
| final `norm` | input (r_final), output (h_final) | LM-head binding; captured as layer group `num_layers` |
| `lm_head` (or tied embedding matmul) | output pre-cap logits (bf16 values), upcast to f32 | decode |

Each tensor is captured once and used in every role it has: `o` is both the Wo output and the `post_attention_layernorm` input, `d` both the Wdown output and the `post_feedforward_layernorm` input, the next layer's `r_in` (or `r_final`) is the layer output. Per position and layer the leaf order is `r_in x_attn q k v q_n k_n v_n a o o_n r_mid x_ffn g u h d d_n` (no `v` on global layers), and `r_final h_final` for the final group.

Per position the capture is a dict `{layer: {name: tensor}}` in the model dtype, plus `logits_precap` (f32), the sampler witness, and the token. Prefill captures `[n_prompt, dim]` slices once; each decode step captures `[1, dim]`. Positions are indexed over the full sequence (prompt then generated).

Attention module internals (RoPE application, softmax) are not hooked. Q and K after RoPE are recomputed by the verifier from the captured post-norm tensors and the public RoPE parameters. The HF rotary code may be imported by the verifier to produce cos and sin tables; this is public configuration, not model state.

Retained-state size (bf16, 31B, per position): roughly 0.17 MB per layer, about 10 MB per position across 60 layers, plus 1 MiB of f32 logits. Suitable for demonstration contexts (a few hundred positions). Tier 2 replaces this with CommitLLM's compact INT8 retained state.

## 6. Commitments

Canonical tensor bytes: `dtype_tag || ndim || shape (u32 LE each) || raw little-endian bytes` (bf16 as uint16). Hash: SHA-256 with domain separation: `H(tag || payload)` where tag is a fixed ASCII label (`"vg/tensor"`, `"vg/layer"`, `"vg/pos"`, `"vg/node"`, `"vg/leaf"`, `"vg/io"`, `"vg/seed"`, `"vg/receipt"`).

- Layer leaf: `H("vg/layer" || layer_index || pos || concat(H(tensor) for each captured name in fixed order))`; the final norm group is layer `num_layers`.
- Position leaf: `H("vg/pos" || pos || input_token || body)` with `body = H("vg/pos_body" || concat(layer leaves) || H(logits_precap) || sampled_token (i32) || H(sampler witness))`; prompt positions that sample nothing use the zero digest and -1. The two levels let an opening bind the input token of every position (a token-only entry: token, body, proof) without opening anything else. A forward position `p` samples token `p + 1`; positions run over `[0, n_prompt + n_gen - 1)` because the last sampled token is never fed back.
- Hash framing: `H(tag, parts) = SHA-256(len(tag) || tag || parts)`; JSON is canonical (`sort_keys`, compact separators).
- Trace root: Merkle root over position leaves (power-of-two padding with a fixed empty-leaf hash).
- Embedding root: Merkle root over `H("vg/tensor" || row_i)` for all vocabulary rows, computed once at keygen.
- Weights root: Merkle root over `H(name || canonical bytes)` of every tensor in the safetensors files, sorted by name.
- IO chain: `c_0 = H("vg/io" || prompt_hash)`, `c_t = H("vg/io" || c_{t-1} || token_t || H(logits_precap_t))`; `io_chain_head = c_{n_gen}`.
- Seed commitment: with a client nonce (the default for `vg chat`) `seed = H("vg/seed_client" || nonce)`, so the prover cannot grind seeds; otherwise `seed = H("vg/seed" || prover_secret || request_id)`. `seed_commitment = H("vg/seed_commit" || seed || request_id)`; `seed` is revealed in the opening and the nonce is in the receipt.
- Prompt hash: `H(token_ids after chat template)`; chat template hash and tokenizer hash from the canonical tokenizer JSON.

Receipt (JSON, version 1):

```json
{
  "version": 1,
  "request_id": "r_…",
  "model": {"id": "google/gemma-4-12B-it", "revision": "…", "weights_root": "…", "config_hash": "…"},
  "manifest": {
    "dtype": "bfloat16", "attn_implementation": "sdpa",
    "temperature": 1.0, "top_p": 0.95, "top_k": 64, "greedy": false,
    "final_logit_softcapping": 30.0, "embed_scale_bf16": 62.0, "rms_norm_eps": 1e-6,
    "sliding_window": 1024, "layer_types_hash": "…", "attention_k_eq_v": true,
    "rope_hash": "…", "qk_norm": true, "thinking": false,
    "chat_template_hash": "…", "tokenizer_hash": "…",
    "speculative": "none", "prefix_caching": false,
    "max_new_tokens": 128, "eos_token_ids": [1, 50, 106],
    "sampler": "vg-sampler-1", "softcap_impl": "vg-canon-softcap-1", "lm_head": "f32"
  },
  "prompt_hash": "…", "seed_commitment": "…", "client_nonce": "…",
  "n_prompt": 57, "n_gen": 184,
  "trace_root": "…", "io_chain_head": "…",
  "prover": {"id": "…", "signature": "ed25519:…"}
}
```

The receipt hash is `H("vg/receipt" || canonical JSON)`; the signature covers it.

## 7. Key generation

Inputs: checkpoint directory (safetensors plus config and tokenizer). Outputs: `key.npz` (secret) and `public.json`.

1. Validate support via `profile.py`; compute `config_hash` over the canonical text config subset (all fields in section 2 plus dims).
2. Weights root over all tensors.
3. Embedding root over vocabulary rows; record `embed_scale_bf16`.
4. For each layer and each matrix family present in that layer type, draw a secret vector `r` of entries in {-1, +1} of length `out_features` from a seeded CSPRNG, and precompute `v = r^T W` in float64 (W upcast from bf16). Store `r` as int8 and `v` as float64. Key size per layer is about the sum of `in_features` over families in float64 (for 31B about 1 MB per layer; about 60 MB total).
5. LM head: `r_lm` of length `vocab_size`, `v_lm = r_lm^T E` in float64 (length hidden).
6. Norm weights: `input_layernorm`, `post_attention_layernorm`, `pre_feedforward_layernorm`, `post_feedforward_layernorm`, `q_norm`, `k_norm` per layer, final `norm`.
7. Public params: `weights_root`, `embedding_root`, `config_hash`, `profile` summary. Everything else in the key is secret; the provider must never see `r` or `v`.

Keygen streams tensors with `safetensors` memory mapping; peak memory is one matrix in float64.

## 8. Prover: decode loop and retained state

- Build the prompt with `apply_chat_template` (thinking off unless requested), tokenize, compute `prompt_hash`.
- Prefill: one forward with hooks on; capture all positions.
- Decode: own loop with `DynamicCache`; each step one forward of the last token; hooks capture; `logits_precap = f32(E) @ f32(h_final)` computed by the engine in true f32 (TF32 off; the head served by the model, upcast once), moved to CPU. An f32 head keeps the LM-head binding tight enough that no single logit can be steered (see section 11).
- Soft-cap and sampling on CPU in f32/f64 with the shared sampler; the witness records the uniform draw(s) and the post-cap logits hash.
- Stop at EOS or `max_new_tokens`.
- Commit: compute layer leaves, position leaves, trace root, IO chain; write retained state to disk under `request_id` (safetensors per position or per layer); build and sign the receipt.
- Retained state TTL and a maximum number of retained requests are configurable.

Tamper modes (`--tamper`), for demonstration only and loudly logged:

| Mode | Effect | Expected reason code |
|---|---|---|
| `weights` | Adds a rank-8 delta (scaled to a configurable relative norm, default 3 percent) to `mlp.down_proj` of one layer at load; receipt still reports the pristine weights root | `FREIVALDS_WDOWN` |
| `identity` | Serves another checkpoint (or the same checkpoint with a different layer's weights swapped) while reporting the declared root | `WEIGHTS_ROOT` if the prover is honest about the root, else `FREIVALDS_*` |
| `sampling` | Samples at a different temperature or top_k than the manifest declares | `DECODE_SAMPLING` |
| `softcap` | Skips the soft-cap before sampling | `DECODE_SOFTCAP` |

## 9. Sampler (shared code)

```
def sample(logits_postcap_f32, temperature, top_k, top_p, rng) -> (token, witness):
    z = logits.astype(float64)
    if greedy: return argmax(z)
    z = z / temperature
    if top_k: keep the top_k largest, others -> -inf
    p = softmax(z)
    if top_p < 1: sort descending, keep smallest prefix with cumulative >= top_p, renormalise
    u = uniform(seed, t)            # (int(H("vg/u" || seed || t)[:8], LE) >> 11) / 2^53
    token = inverse CDF over the kept set in index order
    witness = {"u": u, "step": t}
```

Determinism rules: float64 everywhere after the f32 input; fixed tie-breaking (lowest index, stable sorts); `exp` is the canonical `canon.exp_f64` (Cody-Waite reduction plus a degree-13 polynomial, basic IEEE operations only) and cumulative sums are sequential, so no libm or numpy-version dependence; one uniform per step derived from the revealed seed by SHA-256 (portable across numpy versions, unlike a `default_rng` stream). The soft-cap is likewise canonical (`canon.softcap`, f32 in, f32 out, `tanh` from `exp_f64`) and applied by the prover on CPU. The verifier runs these functions on the opened logits and must get the committed token.

## 10. Verifier

Inputs: receipt, opening, key, public params. No GPU, no weights.

Order of checks (fail fast, reason code on first failure, coverage table always printed). As implemented: receipt schema and signature, opening schema, roots, manifest, seed, prompt, IO chain; then wiring (names and shapes of every opened layer group) and completeness; then the Merkle proof of every opened position (challenged positions first, then K/V provenance rows); then the per-position semantic checks below.

Reason codes: `RECEIPT_SCHEMA`, `RECEIPT_SIGNATURE`, `OPENING_SCHEMA`, `WEIGHTS_ROOT`, `CONFIG_HASH`, `MANIFEST_UNSUPPORTED`, `MANIFEST_MISMATCH`, `SEED_COMMITMENT`, `PROMPT_BINDING`, `IO_CHAIN`, `MERKLE_POSITION`, `EMBEDDING`, `FREIVALDS_WQ|WK|WV|WO|WGATE|WUP|WDOWN`, `BRIDGE_NORM_INPUT|POST_ATTN|PRE_FFN|POST_FFN|Q|K|V|FINAL`, `BRIDGE_GELU`, `BRIDGE_RESIDUAL`, `WIRING`, `KV_PROVENANCE`, `ATTN_REPLAY`, `KV_SHARED`, `LMHEAD_BINDING`, `DECODE_SOFTCAP`, `DECODE_SAMPLING`. Added to the draft list: `OPENING_SCHEMA` (opening malformed, for another request, or not answering the challenge, including withheld challenged tensors or logits), `MANIFEST_MISMATCH` (soft-cap, embed scale, eps, EOS ids, tokenizer or chat-template hash differ from the checkpoint), `BRIDGE_NORM_FINAL` (final norm replay).

1. `RECEIPT_SIGNATURE`, `RECEIPT_SCHEMA`.
2. `WEIGHTS_ROOT`, `CONFIG_HASH`: receipt values equal public params.
3. `MANIFEST_UNSUPPORTED`: speculative off, prefix caching off, attention implementation in the supported set (`sdpa`, `eager`), dtype bf16, sampler and soft-cap implementation ids known. `MANIFEST_MISMATCH`: model-derived fields equal the checkpoint's. `WIRING`: `sliding_window`, `layer_types_hash`, `attention_k_eq_v`, `rope_hash`, `qk_norm` equal the profile's.
4. For each challenged position `p`:
   1. `MERKLE_POSITION`: recompute the position leaf from opened tensors and the provided sibling layer leaves; verify the proof to `trace_root`.
   2. `IO_CHAIN` (generated positions): the opening carries the whole transcript `[(token_t, H(logits_precap_t))]`; the verifier recomputes the chain to `io_chain_head`, checks `len == n_gen`, the stop rule (EOS only as the last token, otherwise `n_gen == max_new_tokens`), and that every opened position's tokens and logits hash agree with it.
   3. `EMBEDDING` (position 0 of a challenge, or any position where the row is opened): Merkle proof of the row to `embedding_root`; `r_in[layer 0] == bf16(row * embed_scale_bf16)` exactly.
   4. For each challenged layer `l`:
      - `FREIVALDS_WQ|WK|WV|WO|WGATE|WUP|WDOWN`: `|r·y - v·x| <= tau(y)` (section 11).
      - `BRIDGE_NORM_*`: `rmsnorm_gemma(input) ≈ captured output` for `input_layernorm`, `post_attention_layernorm`, `pre_feedforward_layernorm`, `post_feedforward_layernorm`, `q_norm`, `k_norm`, `v_norm` (no weight).
      - `BRIDGE_GELU`: `gelu_tanh(g) * u ≈ h`.
      - `BRIDGE_RESIDUAL` (exact): `r_mid == bf16(r_in + post_attention_out)` and `r_in[l+1] == bf16(bf16(r_mid + post_feedforward_out) * layer_scalar)` (or the `norm` input for the last layer), with `layer_scalar` from the key; a bf16 add or product with f32 opmath is replayed bit for bit.
      - `WIRING`: shapes of q, k, v match the layer type (head_dim 256 or 512, KV head counts), `v_proj` absent on global layers, `rope_hash` matches, `qk_norm` present.
      - `KV_PROVENANCE` (attention audit): opened `k_n`, `v_n` rows for the attention window of this layer at position `p` each come with their position-leaf proof (or the whole leaf set if the context is short).
      - `ATTN_REPLAY` (every challenged position): recompute RoPE on `q_n` and `k_n` rows (partial 0.25 on global layers, full on sliding) with the rounding of the bf16 forward (bf16 cos/sin, three roundings; bit-exact against transformers), scores `q·k^T` (scale 1.0), causal and window mask (`kv > q - W`), softmax and `weights @ v_n` with the rounding of the declared `attn_implementation` (`sdpa`: f32 scores, bf16 probabilities for `P V`, f32 normalisation; `eager`: bf16 scores and weights), compare to captured `a`: per head `||a - ref|| / (||ref|| + 2^-14 sqrt(d)) <= 2^-5` (honest worst 0.0075 on Gemma 4 12B / A100). Reported as audited.
      - `KV_SHARED` (global layers): `v_n` within 1 bf16 ulp of `bf16(rmsnorm_noweight(k_pre))` (the f32 reduction order of the mean differs between devices, so bit equality is not portable; 0 ulps observed on CPU).
   5. `LMHEAD_BINDING`: `|r_lm · logits_precap - v_lm · h_final| <= tau`.
   6. `DECODE_SOFTCAP`: the verifier computes `softcap(logits_precap)` with the canonical f32 function and requires its hash to equal the `postcap` hash in the committed sampler witness (post-cap logits are not sent).
   7. `DECODE_SAMPLING`: shared sampler on opened post-cap logits with the revealed seed reproduces the committed token; greedy if the manifest says greedy.
5. `SEED_COMMITMENT`: `H(seed || request_id) == seed_commitment`, and `seed == H(nonce)` when the receipt carries a client nonce.
6. `PROMPT_BINDING`: the prompt tokens are always opened and their hash equals `prompt_hash`.

Challenges are per-position audits `{pos, layers, attention}` (docs/DECISIONS.md 37). Each audited position gets the embedding check, a full audit of its own independently drawn layers, the attention audit if flagged, and the decode checks if it samples a token. By default these are 3 full audits (routine layers plus attention) and, for every other generated token, a decode audit with one random layer. Every other position is opened token-only. A residual stream forged at one layer boundary at every position escapes with `prod_a (1 - |layers_a| / L)`; a fake attention output at one layer escapes with the same product over the audits that replay attention. Both are reported in the verdict.

Hardening after the soundness review (docs/DECISIONS.md 26 to 41): the challenge is an input of the verifier (the auditor's, drawn at random after the receipt), never the prover's echo; decode checks (steps 4.5 to 4.7 plus the final norm) run at every audited position that samples a token, by default every generated token; `verify(expected=...)` binds the receipt to the client's request (prompt, policy, `max_new_tokens`, thinking, nonce); every opened tensor must be finite and every tolerance check requires a finite deviation and bound; a layer group is either a committed leaf or opened tensors, never both.

Verdict JSON:

```json
{"result": "FAIL", "reason": "FREIVALDS_WDOWN", "layer": 41, "position": 98,
 "detail": {"deviation": 3.1e2, "tolerance": 4.7e0},
 "coverage": {"embedding": "verified", "shell": "verified (7/7 families, 10/60 layers)", "bridge": "verified",
              "attention": "audited (last token replay, provenance, wiring)", "decode": "verified", "bindings": "verified"},
 "payload_bytes": 12611540, "verify_ms": 184.0, "positions": [17, 98, 151], "layers": "routine:10"}
```

## 11. Freivalds with a bf16 tolerance (Tier 1)

**As implemented** (supersedes the single-vector L1 bound below, which cannot detect the section 8 weights tamper: with one ±1 vector `|r·Δy| ~ ||Δy||_2` while `2^-7 ||y||_1 ~ 2^-7 sqrt(m) ||y||_2`, so a 3 % change is below the bound for any `m > 32`, e.g. 0.39 vs 0.03 at m = 3840):

- `k` independent secret Rademacher vectors per family (default 16, SHAKE-256 keyed by a secret seed), `v_j = r_j^T W` in float64.
- Statistic `T = sqrt(mean_j (r_j·y - v_j·x)^2)`. The rounding error `e = y - Wx` is independent of the secret `r`, so `E[T^2] = ||e||_2^2 <= (u ||y||_2)^2` with `u = 2^-8`.
- Bound `tau = (3u + 8 sqrt(K) 2^-24) ||y||_2` (three times the worst-case rounding norm plus an f32 accumulation allowance for inner dimension K). An honest prover exceeds it only if a chi-square(k) variable exceeds 9k: below 1e-20 for k = 16 even with every element at half an ulp. Measured honest `T/tau` on tiny CPU: 21 to 25 %.
- Detection: a change moving `y` by `Δ` gives `T ~ ||Δ||_2`, so changes above about 1.2 % of `||y||_2` fail with overwhelming probability; the 3 % rank-8 tamper is rejected on every layer of the tiny model (observed `T/tau` about 2.4).
- The LM-head binding uses the same statistic with `y = logits_precap` (bf16 values from the GPU LM head, so `u = 2^-8`, `K = hidden`).

Draft text kept for reference:

For a projection `y = W x` computed on GPU in bf16 inputs with f32 accumulation and bf16 output rounding, and verifier-side float64 arithmetic with `r` in {-1,+1}^m and `v = r^T W` precomputed in float64:

- Exact identity: `r·(W x) = v·x`.
- Observed: `r·y` where `y = round_bf16(W x + accumulation error)`.
- Elementwise `|y_i - (Wx)_i| <= u |(Wx)_i| + e_acc`, with bf16 unit roundoff `u = 2^-8` and `e_acc` the f32 accumulation error, bounded by `K * 2^-24 * sum_j |W_ij x_j|` which is negligible against the bf16 term for `K <= 21504`.
- Therefore `|r·y - v·x| <= u * sum_i |y_i| * (1 + O(u)) + m * e_acc`.
- Tolerance used: `tau(y) = 2^-7 * ||y||_1 + 1e-6 * ||y||_1` (a factor-two margin on the rounding bound plus the accumulation term).

The verdict prints both the deviation and `tau` for every family so the margin is visible. Detection power: a weight change that moves `y` by more than about 1 percent of its L1 norm in the direction of `r` fails; smaller changes may pass in Tier 1. Multiple independent `r` vectors per family (configurable, default 1) reduce the chance that a change is orthogonal to `r`.

Tier 2 replaces this with CommitLLM's exact INT8 field check (soundness error about 2^-32 per check, no tolerance).

The LM-head binding uses the same bound with `y = logits_precap` computed in f32 (`u = 2^-24`; the products of bf16 values are exact in f32, so the bound is dominated by the accumulation term over `K = hidden`). As implemented this is the RMS statistic of this section with `u = 2^-24`: about `3e-5 ||y||_2` for K = 3840, so one logit can move by at most about 0.015 logit RMS on Gemma 4 (with bf16 logits the same L2 bound allowed about 6 RMS, enough to choose the token).

## 12. Tiny mode

`vg tiny` builds a `Gemma4TextConfig` (or the text config class the installed `transformers` exposes) by copying the real 12B text config and shrinking: `hidden_size 64, num_hidden_layers 6, num_attention_heads 4, num_key_value_heads 2, head_dim 16, global_head_dim 32, num_global_key_value_heads 1, intermediate_size 128, vocab_size 512, sliding_window 8, layer_types = [sliding x5, full]`, keeping `attention_k_eq_v`, `final_logit_softcapping 30.0`, `rope_parameters`, `tie_word_embeddings`, `rms_norm_eps`. Weights are randomly initialised with a fixed seed and saved as a checkpoint directory so keygen, serve, audit and verify run unchanged on CPU. The tokenizer is a byte-level stand-in for tests. Every test in the suite runs in tiny mode; the real-model run is a smoke test on top.

## 13. Interfaces

CLI (all via `uv run`; on Modal the same commands run inside the functions in `infra/modal_app.py`):

```
vg tiny     --out ./tiny-gemma4                               # build the tiny checkpoint
vg keygen   --model <dir|hf-id> --out ./keys/<name>            # key.npz (secret) + public.json
vg serve    --model <dir|hf-id> --public ./keys/<name>/public.json --port 8000 [--tamper weights|identity|sampling|softcap] [--device cuda|cpu]
vg chat     --server http://localhost:8000 --model <dir|hf-id> --prompt "…" [--max-new-tokens 128] [--thinking]   # receipt.json + request.json (prompt tokens, nonce, prover id)
vg audit    --server … --receipt receipt.json [--positions random:3] [--layers routine|full|<list>] [--decode all-gen] [--decode-layers 1] --out opening.bin   # + challenge.json
vg verify   --key ./keys/<name>/key.npz --public … --receipt receipt.json --opening opening.bin --challenge challenge.json [--request request.json] [--deep]
vg demo     --model … [--tiny] [--device …]                    # honest PASS, four tampers FAIL, cost summary
vg bench    --model … [--tiny]                                 # overhead, retained bytes, opening bytes, verify ms
```

HTTP:

- `POST /chat {messages, max_new_tokens, thinking, sampling overrides?, nonce?}` -> `{text, token_ids, receipt, timings}`
- `POST /audit {request_id, audits: [{pos, layers, attention}], tier}` -> opening (binary: safetensors blob plus JSON index)
- `GET /health` -> model id, weights root, config hash, prover id, retained requests count

Opening layout: `index.json` (positions, layers, which tensors, proofs) and `tensors.safetensors` (opened tensors keyed `p{pos}/l{layer}/{name}`, logits keyed `p{pos}/logits_precap` and `p{pos}/logits_postcap`), the revealed seed and witnesses, sibling layer leaves per position, Merkle paths.

## 13a. Packaging and environments

- `uv` project, `src/` layout, Python 3.12 pinned in `.python-version`, `uv.lock` committed. Console script `vg` from `vgemma.cli:app` (Typer).
- Extras: `gpu` (CUDA torch wheel via a uv index with marker-based selection), `modal`. Dependency group `dev` (pytest, pytest-timeout, ruff).
- The default `uv sync` yields a CPU-only environment that runs tiny mode and the full test suite; `uv sync --extra gpu` yields the CUDA environment used in Modal images and on vast.ai.
- All entry points, scripts, CI steps and Modal image builds use `uv run` and `uv sync --locked`. No other package manager appears in the repository.

## 13b. GPU infrastructure

Primary: Modal, driven from `infra/modal_app.py`.

| Function | Resources | Mounts | Purpose |
|---|---|---|---|
| `download` | CPU | `/cache` | Pre-pull a model into `HF_HOME=/cache/hf` |
| `tests_tiny` | CPU | none | `uv run pytest` in the image, same code as the GPU functions |
| `keygen` | CPU, 64 GB RAM, long timeout | `/cache`, `/state`, `/keys` | Writes `key.npz` to `/keys/<name>/` and `public.json` to `/state/<name>/` |
| `serve` | GPU (A100-40GB or L40S for 12B; A100-80GB or H100 for 31B), `@modal.web_server(8000)` | `/cache`, `/state` | The provider; never mounts `/keys` |
| `demo` | GPU as above | `/cache`, `/state`, `/keys` | Starts the server as a subprocess with only `/state` visible to it, then runs chat, audit and verify in the parent with the key; prints the story and the cost summary |
| `bench` | GPU as above | `/cache`, `/state`, `/keys` | Overhead, retained bytes, opening bytes, verify ms, keygen time |

Image: `modal.Image.debian_slim(python_version="3.12").uv_sync(extras=["gpu"])` from the repo's `pyproject.toml` and `uv.lock`, plus `add_local_python_source("vgemma")`. Volumes: `vg-hf-cache`, `vg-state`, `vg-keys`. The key never enters the serve process: this is enforced by which volumes each function mounts.

Backup: vast.ai, driven from `infra/vast/setup.sh` over SSH on a rented single-GPU instance of the same class (CUDA 12 PyTorch image, 200 GB disk). The script installs `uv`, clones the repo, runs `uv sync --extra gpu`, sets `HF_HOME` on the instance disk, and the same `uv run vg ...` commands apply. Only the launcher differs between the two providers; the code path is identical.

## 14. Tests

- Unit: canonical functions against `transformers` modules on random inputs (`rmsnorm_gemma` with and without weight, `gelu_tanh`, `softcap`, RoPE tables for both layer types, embed scale rounding); Merkle proofs; sampler determinism across processes.
- Protocol, tiny mode: honest run passes with full and routine layer sets, greedy and sampled, EOS and max-token stops, multi-position challenges.
- Adversarial, tiny mode: four tamper modes; opened-tensor bit flips in each family; swapped layer; swapped position; wrong Merkle sibling; wrong seed; wrong manifest field; removed `v_norm`; full RoPE on a global layer (verifier-side wiring check); each rejected with the expected reason code.
- Real model smoke: `vg demo` on the configured GPU model.
- Benchmarks: `vg bench` records tokens per second with hooks on and off, retained bytes per position, opening bytes per challenged position, verify ms per position, keygen wall time, key size.

## 15. Tier 2: exact INT8 path in CommitLLM

Goal: the same receipts and verdict semantics with exact linear and bridge checks, on a vLLM serving path.

1. Derived checkpoint: W8A8 (INT8 symmetric per-channel weights; activation scheme identical to the neuralmagic Llama-3.1-8B W8A8 checkpoint CommitLLM is validated on) produced with `llm-compressor` from `google/gemma-4-31B-it` or `12B-it`, text decoder only, embedding and tied head kept in bf16, pinned recipe, calibration set and seed recorded, hash published.
2. CommitLLM changes (new profile `gemma4-w8a8`):
   - keygen: tensor map (tied head fallback to `embed_tokens`; `pre_feedforward_layernorm` takes the FFN pre-norm slot; `post_attention_layernorm` and `post_feedforward_layernorm` are new post-norm slots; `q_norm` and `k_norm` weights), per-layer-type shapes, `rope_parameters` and `layer_types` in the config hash, decode artefact from the tied embedding (2.82 GB for 31B).
   - verilm-core: `gelu_tanh`, `rmsnorm_gemma` (f32-faithful, optional weight), `softcap`, partial RoPE tables; attention wiring and KV provenance for layer types, window eviction, shared K and V.
   - sidecar: Gemma family detection, hook placement on vLLM's packed `qkv_proj` and `gate_up_proj`, pre-cap logit capture from the logits processor, manifest fields, fail-closed checks; speculative decoding and prefix caching disabled.
   - Captured-boundary path for `x_attn` and `x_ffn` (post-quant i8 captured on GPU) because vLLM's fused norm-quant kernel is not replayable from the bridge (observed on Qwen); the norm boundary is audited to ±1 LSB.
3. Stack: current vLLM, compressed-tensors, llm-compressor and transformers releases; CommitLLM's pins are from April 2025 and the sidecar must be re-validated on Llama first.
4. Expected numbers for 31B: retained logits 1 MiB per token; opening about 1.2 MiB per challenged token; verifier about 1 to 3 ms per challenged token; keygen minutes; tracing overhead in the range CommitLLM measured.

## 16. Attribution and licensing

- Protocol design, terminology (commit, audit, opening, routine and full audit, CapturedLogits, bridge), receipt structure and the verified / audited / open claim boundary follow CommitLLM (LambdaClass, MIT). NOTICE lists the upstream repository and commit.
- Any code copied from CommitLLM keeps its MIT header and is listed in NOTICE with source path and destination path.
- Gemma 4 weights are Apache 2.0; derived checkpoints carry the licence and attribution.
- This repository is MIT.

## 17. References

- lambdaclass/CommitLLM: README, roadmap, docs/design/deterministic-attention-spec.md, docs/security/redteam_audit_only.md, sidecar/verilm, crates/verilm-keygen.
- google/gemma-4-31B-it, google/gemma-4-12B-it: config.json and model cards; Gemma 4 Technical Report (arXiv 2607.02770); Gemma documentation (ai.google.dev/gemma/docs).
- huggingface/transformers: src/transformers/models/gemma4/modeling_gemma4.py.
- vllm-project/vllm: vllm/model_executor/models/gemma4.py, registry.py.
- R. Freivalds, Probabilistic machines can use less running time (1977).
