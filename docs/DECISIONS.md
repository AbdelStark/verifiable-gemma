# Decisions

One line per decision, with the reason. Newest last.

1. **Freivalds: 16 secret vectors and an L2 bound** (`tau = (3·2^-8 + 8·sqrt(K)·2^-24)·||y||_2` on the RMS of the 16 projections) instead of one vector with `2^-7·||y||_1`. The L1 bound is so loose that the spec's 3 % weights tamper passes it on every real layer: a ±1 projection of a change scales with `||Δy||_2`, the bound with `sqrt(m)·||y||_2`. The L2 bound holds with high probability over the verifier's secret `r` (TECH_SPEC section 11).
2. **`layer_scalar` is replayed, not asserted to be 1.** The published 12B checkpoint stores bf16 values from 0.0045 to 0.76 (read with HTTP range requests). The bridge replays `bf16(bf16(r_mid + d_n) * s)` with `s` from the key, and the tiny checkpoint uses random `s` in [0.3, 1.0] to exercise it.
3. **Residual chain and scaled embedding are exact checks.** A bf16 add (or a bf16 × bf16 product) computed with f32 opmath is reproducible bit for bit with numpy f32 and round-to-nearest-even; tested against torch on 100k values.
4. **`KV_SHARED` allows 1 bf16 ulp.** The f32 mean in `v_norm` is reduced in a device-specific order, so bit equality between GPU and CPU replay is not guaranteed. 0 ulps are observed on CPU.
5. **Sampler randomness comes from `H("vg/u" || seed || step)`, not `numpy.default_rng`.** This keeps it portable across numpy versions and platforms, and the verifier recomputes `u` from the revealed seed.
6. **Canonical `exp`, `tanh` and soft-cap are built from basic IEEE operations only.** libm and numpy SIMD `tanh`/`exp` differ by an ulp across platforms, which breaks bit-exact `DECODE_SOFTCAP`. The prover applies the soft-cap on CPU from the captured pre-cap logits (cross-process test).
7. **RoPE replay is bf16-faithful.** cos and sin are cast to bf16, and `x·cos` and `rot(x)·sin` are rounded before the add, as in the eager forward. This is bit-exact against `transformers` on CPU and shrinks the attention-replay noise.
8. **Attention replay runs at every challenged position, prefill included,** with a per-head relative L2 bound of 2^-5. The worst honest value on tiny CPU (SDPA and eager) is 0.0027, 8 to 11 % of the bound. It must be re-measured on GPU with `vg bench --calibrate` and reported. It stays "audited".
9. **Bridge bounds are elementwise:** norms `|y-ref| <= 2^-7|ref| + 2^-126` (one rounding, factor-two margin); GELU gate `<= 2^-6|ref| + 2^-20|g·u| + 2^-126` (two roundings; the middle term covers f32 cancellation in `1 + tanh` for very negative inputs). Honest worst case: 50 % and 45 % of the bound.
10. **Reason codes added:** `OPENING_SCHEMA`, `MANIFEST_MISMATCH`, `BRIDGE_NORM_FINAL`, and named `BRIDGE_NORM_INPUT|POST_ATTN|PRE_FFN|POST_FFN|Q|K|V`.
11. **Bit flips in opened tensors fail `MERKLE_POSITION`,** because every opened tensor is committed. The semantic codes (`FREIVALDS_*`, `BRIDGE_*`, `DECODE_*`, ...) are exercised by a prover that commits to a modified trace, or that serves a consistently modified model.
12. **Check order:** wiring (names and shapes per layer type) runs before Merkle, challenged positions before K/V provenance rows, and seed, prompt and IO bindings before the per-position checks. A wrong seed therefore reports `SEED_COMMITMENT`, not `DECODE_SAMPLING`.
13. **The position leaf includes the input token,** and the final norm is layer group `num_layers`. The seed commitment uses its own tag (`vg/seed_commit`).
14. **The manifest gained `max_new_tokens`, `eos_token_ids`, `sampler` and `softcap_impl`.** The IO chain check enforces the stop rule (EOS only at the end, otherwise exactly `max_new_tokens`), which catches truncation.
15. **`public.json` records the EOS ids, tokenizer hash and chat-template hash** computed at keygen, so the manifest values are bound to the checkpoint (`MANIFEST_MISMATCH`).
16. ~~LM-head logits are the bf16 `lm_head` output upcast to f32.~~ Superseded by 30: the prover computes the logits in f32.
17. **The embedding row is opened with a Merkle proof;** the verifier holds only the root (PRD default).
18. **Receipts are signed with Ed25519 by default.** The prover identity lives in the server state directory, and the auditor pins the prover id (the demo pins it on first contact).
19. **Tamper `identity` swaps two sliding layers in memory** and reports the declared root, so it fails `FREIVALDS_WQ`. `identity-root` reports the root of what it serves, so it fails `WEIGHTS_ROOT`. The extra modes `rope-full` and `no-vnorm` exist for the adversarial suite.
20. **The demo audits tamper runs on all layers.** A routine subset only covers the layers it opens; `test_weights_tamper_missed_when_layer_not_challenged` states this.
21. **Torch index layout:** plain `uv sync` gets torch from PyPI (CPU build on macOS); on Linux, `--extra cpu` gets the CPU-only wheel and `--extra gpu` the CUDA 12.9 wheel (torch 2.13, the newest `cu12x` build). uv cannot make an index conditional on the absence of an extra, so the Linux default is PyPI's wheel.
22. **The 12B checkpoint is `gemma4_unified` in transformers 5.18.** Its decoder code is identical to `gemma4`; both `gemma4_text` and `gemma4_unified_text` are supported, and modules are discovered by name pattern.
23. **The tiny tokenizer is a byte-level stand-in with its own `vg_tokenizer.json`.** The real models use the checkpoint's `tokenizer.json` and chat template (thinking off unless requested).
24. **Opening container:** `VGOPEN01 || u64 len || index JSON || safetensors` (bf16 stored as U16). The verifier needs only numpy and safetensors to read it.
25. **The verifier imports no torch** (numpy only). The server reads only `public.json` and refuses a `.npz` path.

## After the adversarial soundness review (2026-10-01)

An independent review of the first version found six ways to get a PASS outside the claim boundary. Each one was confirmed with a PoC, then fixed and pinned by a test in `tests/test_adversarial_review.py`.

26. **Non-finite values fail closed.** Opened tensors must be finite (bf16 exponent 0xFF, or non-finite f32, is rejected), and a tolerance check fails unless both the deviation and the bound are finite. Before: a `+inf` logit made `inf <= inf` pass and forced any token.
27. **A layer group carries either a `leaf` or `tensors`, never both.** Before: the committed leaf satisfied the Merkle check while fabricated tensors fed the semantic checks.
28. **Challenge positions are drawn by the auditor at random by default** (`random:3`, `SystemRandom`). Before: `auto` always opened the first, middle and last token, so every other token could be forged. `edges` remains for reproducible tests only.
29. **Decode checks run at every generated token by default** (first as `decode_positions`; superseded by the per-position audits of 37). They cover the LM-head binding, the soft-cap and the sampled token, plus the final norm. The opening carries only the pre-cap logits, about 1 MiB per token for Gemma 4; the verifier recomputes the post-cap logits and checks them against the sampler witness.
30. **The prover computes the LM head in f32** (`E` upcast once, TF32 disabled), and the binding uses the f32 bound. Before: with bf16 logits the L2 bound let one logit move by about `3·2^-8·‖z‖₂`, which is roughly 6 logit RMS over a 262k vocabulary, enough to choose the token. Now the freedom is about 0.015 RMS on Gemma 4 and the honest deviation is 1.3 % of the bound on tiny. The cost is an extra f32 copy of the head (4 GB on 12B).
31. **The opening always reveals the prompt,** and `verify(expected=...)` binds the receipt to what the client sent: prompt tokens or hash, sampling policy, `max_new_tokens`, thinking and nonce. `vg chat` writes `request.json`, and `vg verify --request` uses it and pins the prover id from it. Before: omitting the prompt skipped the binding, and nothing tied the manifest to the request.
32. **A client nonce fixes the sampling seed** (`seed = H("vg/seed_client" || nonce)`). This stops best-of-N grinding over prover seeds. Without a nonce the prover's own seed is used, and grinding remains possible; this is documented.
33. **The challenge is a required input of `verify()`,** the auditor's own and never the prover's echo, and an empty layer set is rejected. `vg audit` writes `challenge.json` and `vg verify` requires it.
34. **Every check runs under its own stage code,** so malformed data fails with the right reason.
35. **Measured weights-tamper margin on tiny** (3 %, rank 8, layer 3, 72 decode positions): deviation/bound min 1.10, median 2.6. Detection is per position and probabilistic; three random positions catch it.
36. **Remaining Tier 1 limit, stated in the README:** a shell output may deviate within the L2 bound, about 1.2 % of `‖y‖₂`, and the deviation may be concentrated on a few elements.

## After the second review pass (2026-10-01)

A second pass confirmed that 26 to 35 close the original holes, and found three more. Each one is now a test in `tests/test_adversarial_review.py` (sections 8 and 9).

37. **Per-position audits replace the shared layer set.** A challenge is a list of audits `{pos, layers, attention}`, and each audit draws its own layers.
    - Every generated token gets a decode audit with one random layer audited in full (no attention), on top of a few full audits with a routine layer subset and attention.
    - Before: a residual stream forged at one layer boundary (for example `r_final`, which lets the honest LM head emit any text) at every token was caught only if the one shared layer set contained the producing layer. That escapes with probability `1 − k/L`, about 79 % on 12B at `routine:10`.
    - Now it escapes with `prod_a (1 − |layers_a|/L)`. The verdict and `vg audit` print it: 6.3e-4 for the default challenge on tiny with 32 tokens. On 12B with 3 full `routine:10` audits it is `0.50 × (47/48)^n`: 0.30 for 24 tokens and 0.033 for 128. `--decode-layers 2` gives 0.18 for 24 tokens, at about 0.2 MB more per token.
38. **The position leaf is `H(pos ‖ input_token ‖ body)`,** and every position not otherwise opened is sent as a token-only entry `{input_token, body, proof}`. Before: input tokens at prompt positions the challenge did not open were not bound to the revealed prompt, which allowed a prompt substitution outside a global layer's window, or always with attention off. The cost is about 36 bytes plus a proof per position.
39. **The client always templates its own prompt** with the checkpoint's public tokenizer: `vg chat --model` is required, and `request.json` always carries the prompt tokens and the tokenizer and template hashes. `verify` fails `PROMPT_BINDING` when it is given a client request without a prompt binding.
40. **The client also pins `attn_implementation`** (read from `/health` before the request), so the prover cannot choose between `sdpa` and `eager` afterwards to pick an output.
41. **Not yet measured on a GPU:** the f32 LM-head bound's `sqrt(K)` accumulation margin (1.3 % of the bound on tiny CPU) and TF32 being off. `vg bench --calibrate` on the real model has to confirm them before the GPU numbers are quoted.

## After the third review pass (2026-10-01)

42. **The verdict reports two escape probabilities.** A residual stream forged at one layer boundary is caught by any audit of the producing layer (`forged_boundary_escape`). A fake attention output at one layer is caught only by audits that replay attention (`fake_attention_escape`). Before: only the first was printed, which overstated how well decode audits (attention off) cover the audited-only attention path. `vg audit --decode-attention` extends the replay to decode audits, at the cost of opening every attended K/V row.
43. **`check_expected` now enforces every client field** (policy, `attn_implementation`, tokenizer and template hashes). The earlier edit had defined the list but left the loop on the policy fields only.
