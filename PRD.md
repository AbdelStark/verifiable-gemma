# verifiable-gemma: Product Requirements

Codename: `verifiable-gemma`
Status: draft v0.1
License: MIT (protocol and code lineage: CommitLLM by LambdaClass, MIT; model: Gemma 4 by Google DeepMind, Apache 2.0)

## 1. One-paragraph summary

verifiable-gemma makes every response from an open-weight Gemma 4 model carry a cryptographic receipt, and lets an auditor who holds a small secret verification key challenge any response afterwards and check, on a CPU, that the declared Gemma checkpoint, deployment configuration and sampling policy produced it. It applies the CommitLLM commit-and-audit protocol (commit at serve time, open only challenged parts, verify large matrix products with randomised Freivalds checks, replay the non-linear parts canonically, verify sampled decode exactly) to Gemma 4, which CommitLLM does not support today. The provider keeps a normal GPU serving path; there is no zero-knowledge prover and no per-response proof generation.

## 2. Problem

When a model is served behind an API, nobody but the operator knows which model answered, with which weights, at which temperature, with which safety configuration. This matters for users who pay for a specific model, for platforms that resell inference, for regulators and auditors who need to know what is deployed, and for any setting where two parties do not trust each other's claims about what is running.

Existing answers are weak or expensive. Fingerprinting and statistical tests give signals, not proof. Zero-knowledge proofs of inference cost orders of magnitude more than the inference. Trusted hardware moves the trust rather than removing it.

Open weights change the economics. If the checkpoint is public, a verifier can build a key from it once, and a provider can commit to its execution cheaply and open only what is challenged. CommitLLM demonstrates this for Llama and Qwen. Gemma 4 is the most capable open-weight family under a permissive licence, and its architecture (sliding and global attention, QK-norm, shared K and V on global layers, GELU-tanh gating, logit soft-cap, tied embeddings) is different enough from Llama that supporting it is real work, not a configuration change.

## 3. Goals

1. Serve Gemma 4 with a standard inference stack and return a compact receipt with every response.
2. Let an auditor challenge any committed response at chosen token positions and layers, receive an opening, and verify it on CPU with a small key, without the model weights.
3. Verify, on the opened data: the embedding lookup and scaling, every linear projection (Q, K, V, O, gate, up, down, LM head), the residual chain through Gemma's four per-layer norms and the GELU gate, the final logit soft-cap, the sampled token under the declared sampling policy, and all bindings (prompt, manifest, seed, token count, IO chain).
4. Audit the attention path: provenance of K and V rows, score recomputation for challenged generated tokens, and the Gemma-specific wiring (layer types, sliding window, per-layer-type head dimension, shared K and V, partial RoPE, QK-norm).
5. Reject tampering live, each with a named reason: modified weights (a merged fine-tune), a different checkpoint behind the declared name, a silently changed sampling policy, a removed soft-cap.
6. Report the costs honestly: serving overhead, retained state per token, opening size per challenged token, verifier time per challenged token.
7. State the claim boundary explicitly, using CommitLLM's own vocabulary: verified, audited, open.

## 4. Non-goals

- Zero-knowledge or publicly verifiable proofs. Verification requires the auditor's secret key.
- Verifying attention outputs at arbitrary prefill positions on stock fused kernels. This is an open problem in CommitLLM and remains audited rather than verified here.
- Closed-weight models, TPU serving, multimodal inputs, speculative decoding, prefix caching across requests, batched multi-tenant serving.
- Mixture-of-experts Gemma 4 (26B-A4B) and the per-layer-embedding small models (E2B, E4B).
- Production key management, revocation, or multi-auditor policy.

## 5. Users

| User | Needs |
|---|---|
| Auditor | A CLI that takes a receipt and an opening and prints PASS or FAIL with a reason code, a coverage table and timings. A key that is small and never shared with the provider. |
| Provider / operator | A serving path with a measured, bounded overhead, a retained-state budget it can plan for, and no change to answer semantics. |
| Integrator | A receipt and verdict format that can be logged, signed and forwarded (for example into a monitoring report or a transparency log). |
| Developer | A clean profile abstraction for the model family, a tiny CPU-only model for tests, and an adversarial suite that proves the verifier rejects for the right reasons. |

## 6. Scope: two tiers

### Tier 1: end-to-end MVP (top priority)

A self-contained Python implementation of the protocol on top of Hugging Face `transformers` with Gemma 4 in bf16 on a single GPU, plus a tiny randomly initialised Gemma 4 configuration that runs the whole pipeline on CPU for tests and development.

- Linear projections are checked with Freivalds against key vectors precomputed from the public weights, with an explicit tolerance derived from bf16 output rounding; the tolerance is printed next to the observed deviation so the margin is visible.
- Non-linear parts (norms, GELU gate, residual adds, soft-cap) are replayed canonically in float64 and compared within a stated tolerance.
- Sampled decode is exact: the prover samples on CPU from the committed logits with committed randomness; the verifier replays the same code and gets the same token or fails.
- Attention for generated tokens is replayed for the challenged position (single-query decode) from opened Q, K, V rows and compared within tolerance; K and V rows are provenance-checked against the commitment. Prefill attention is audited for provenance and wiring only.
- Everything is committed with Merkle trees and hash chains; receipts are small; openings are per challenged position.

### Tier 2: exact INT8 path (after the MVP)

Port the Gemma 4 profile into CommitLLM proper: a W8A8 derived checkpoint, the vLLM sidecar, and the Rust verifier, so that the linear shell and bridges are exact (information-theoretically sound Freivalds over a prime field, canonical INT8 replay) rather than tolerance-bounded. The MVP's manifest fields, tensor map, canonical functions and adversarial suite are designed to carry over unchanged.

## 7. Functional requirements

| ID | Requirement | Tier |
|---|---|---|
| F1 | `vg keygen` builds a verifier key from a public Gemma 4 checkpoint: weights root, embedding Merkle root, Freivalds vectors per weight matrix with per-layer-type shapes, norm and QK-norm weights, canonical config hash. The key is small relative to the model and is secret. | 1 |
| F2 | `vg serve` runs Gemma 4 with capture hooks and exposes `/chat`, `/audit`, `/health`. Each `/chat` returns the text plus a receipt. Retained state is kept for an audit window and then discarded. | 1 |
| F3 | The receipt binds: checkpoint identity (weights root), canonical config hash, deployment manifest (sampling policy, soft-cap, dtype, attention implementation, chat template hash, tokenizer hash, thinking flag), prompt hash, seed commitment, token counts, trace root, IO chain head, prover identity. | 1 |
| F4 | `vg audit` requests an opening for chosen positions and layers; the opening contains only what the verifier needs plus Merkle proofs. | 1 |
| F5 | `vg verify` checks an opening against a receipt and key without the model weights, and prints a verdict JSON: result, reason code, failing component, coverage table, payload bytes, milliseconds. | 1 |
| F6 | Verified components: embedding row and scale; all seven projection families and the LM head (Freivalds, tolerance-bounded in Tier 1, exact in Tier 2); bridge replay (four norms, GELU gate, residual chain); soft-cap replay; exact sampled decode (temperature, top_k, top_p, greedy); prompt, manifest, seed, token-count and IO-chain bindings; Merkle membership of every opened tensor. | 1 |
| F7 | Audited components: K and V row provenance; single-query attention replay for challenged generated tokens; wiring checks for layer types, sliding window, head dimensions, KV head counts, shared K and V on global layers, partial RoPE parameters, QK-norm presence. | 1 |
| F8 | Tamper modes in the server for demonstration: `weights` (a low-rank delta merged into one projection), `identity` (a different checkpoint served behind the declared root), `sampling` (temperature changed while the manifest says otherwise), `softcap` (soft-cap skipped). Each must be rejected with the matching reason code. | 1 |
| F9 | `vg demo` runs the full story end to end: honest request, audit, PASS; then each tamper mode, audit, FAIL with reason; then a cost summary. | 1 |
| F10 | Tiny mode: a small random Gemma 4 text configuration that preserves every architectural feature (layer-type pattern, global head dimension, shared K and V, partial RoPE, QK-norm, soft-cap, tied embeddings, embedding scale) and runs the entire pipeline and test suite on CPU. | 1 |
| F11 | Adversarial test suite: every tamper mode plus field-level tampering of openings (bit flips in opened tensors, swapped layers, swapped positions, wrong Merkle path, wrong seed, wrong manifest) is rejected for the right reason. | 1 |
| F12 | Fail closed: unsupported configurations (MoE block enabled, per-layer inputs, KV-shared layers, double-wide MLP, bidirectional attention beyond vision) are rejected at keygen and at serve. | 1 |
| F13 | Metrics printed by `vg demo` and `vg bench`: tokens per second with and without capture, retained bytes per token, opening bytes per challenged position, verifier milliseconds per challenged position, keygen wall time and key size. | 1 |
| F14 | Receipts and verdicts are JSON with a documented schema and a version field; receipts are optionally signed by the prover (Ed25519) so they can be forwarded. | 1 |
| F16 | Packaging: a `uv` project (`pyproject.toml`, committed `uv.lock`, `src/` layout, `vg` console script); CPU-only install by default, CUDA via an extra. | 1 |
| F17 | GPU runs are code-driven on Modal (`infra/modal_app.py`: download, keygen, serve, demo, bench functions, volumes for model cache, state and keys, with the secret key never mounted into the serving function). vast.ai is the backup launcher with the same commands over SSH. | 1 |
| F15 | W8A8 derived checkpoint with a pinned, reproducible recipe; CommitLLM `gemma4-w8a8` profile for keygen, sidecar and Rust verifier; exact bridge semantics. | 2 |

## 8. Non-functional requirements

- Correctness over speed: a check is either exact, tolerance-bounded with the bound printed, or declared audit-only. No silent tolerance.
- Verifier runs on CPU with no GPU and no model weights; memory bounded by the key plus one opening.
- Serving overhead measured and reported; target at most 25 percent throughput loss with capture on, for the demo prompt sizes.
- Retained state bounded per token and per request; openings bounded per challenged position; both reported.
- Deterministic sampling: the prover's sampler and the verifier's replay are the same code path on CPU with the same seeded generator.
- Reproducible environments: `uv sync --locked` reproduces the exact dependency set locally, in Modal images and on vast.ai.
- Honest documentation: README carries the verified / audited / open table.
- Clean attribution: CommitLLM protocol lineage and any copied code acknowledged in NOTICE with file paths and the upstream commit.

## 9. Demo flow

1. Show the receipt for an honest Gemma 4 response: model identity, manifest, token count, trace root. Say what is committed and that nothing is proved yet.
2. Challenge three generated positions and a routine layer subset. Show the opening size and the verdict: PASS, coverage table, verifier milliseconds.
3. Tamper `weights`: a merged low-rank delta in one down projection. Verdict FAIL, reason `FREIVALDS_WDOWN` with the layer, with the observed deviation against the tolerance.
4. Tamper `identity`: a different Gemma checkpoint behind the declared root. Verdict FAIL, reason `WEIGHTS_ROOT` or `FREIVALDS_*` depending on what the prover claims.
5. Tamper `sampling`: temperature changed. Verdict FAIL, reason `DECODE_SAMPLING`.
6. Tamper `softcap`: soft-cap removed. Verdict FAIL, reason `DECODE_SOFTCAP`.
7. Show the cost summary and the verified / audited / open table.

## 10. Success criteria

- `vg demo --tiny` passes on CPU: honest PASS and four tamper FAILs with correct reason codes.
- `vg demo` passes on a real Gemma 4 checkpoint on one GPU with the same outcomes, launched as a Modal function from the repository.
- Adversarial suite green: every scenario rejected for the right reason.
- Metrics printed and recorded in the README.
- Receipt, opening and verdict schemas documented; NOTICE complete.

## 11. Assumptions and risks

- Tolerance-bounded Freivalds on bf16 can miss weight changes that are smaller than bf16 rounding noise. The tolerance and the observed deviation are both printed; the exact Tier 2 path removes the limitation.
- Attention outputs at prefill positions are not verified; the attention claim is audited, as in CommitLLM.
- Retained state in Tier 1 is large (bf16 tensors per layer per position). It is sized for demonstration contexts and documented; Tier 2 uses CommitLLM's compact retained state.
- Gemma 4 model classes differ between sizes (`Gemma4ForConditionalGeneration`, `Gemma4UnifiedForConditionalGeneration`); the text decoder is the same design. Module discovery is by name pattern, not by class.
- Capturing per-layer tensors through forward hooks assumes the eager or SDPA attention path in `transformers`; fused serving kernels (vLLM) are Tier 2.

## 12. Open questions

- Whether to keep the prover-side LM head on GPU (capture pre-cap logits, upcast to f32) or compute it on CPU for bit-exact logits. Default: GPU, with Freivalds binding of the captured logits to the final hidden state.
- Whether the embedding row is opened with a Merkle proof (verifier holds only the root) or the verifier is allowed read access to the public embedding matrix. Default: Merkle proof.
- Whether to sign receipts by default. Default: yes, Ed25519, key generated at first run.
