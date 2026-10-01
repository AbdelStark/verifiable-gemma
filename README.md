# verifiable-gemma

A commit-and-audit verification layer for Gemma 4 inference, following the
[CommitLLM](https://github.com/lambdaclass/CommitLLM) protocol.

Every response from a Gemma 4 model comes with a signed receipt that commits to the execution trace.
Afterwards an auditor holding a small secret key can challenge chosen token positions and layers,
receive an opening, and check on a CPU, without the model weights, that the declared checkpoint,
deployment configuration and sampling policy produced the answer. There is no zero-knowledge prover
and no per-response proof. The provider runs its normal GPU path and opens only what is challenged.

Status: Tier 1 MVP (Python on `transformers`, bf16, tolerance-bounded linear checks, exact decode).
`vg demo --tiny` passes on CPU, and the 258-test suite runs in about 15 s. On the real model,
`google/gemma-4-12B-it` on one A100-40GB launched from `infra/modal_app.py`, `vg demo` gives an
honest PASS on routine and full audits and the four tamper FAILs with the expected codes; the
GPU metrics below come from those runs. Read [PRD.md](PRD.md) and [TECH_SPEC.md](TECH_SPEC.md) for the
design, [docs/DECISIONS.md](docs/DECISIONS.md) for where the implementation departs from the draft
spec, and [docs/SCHEMAS.md](docs/SCHEMAS.md) for every hashed format.

## Quickstart (CPU, tiny model)

```bash
uv sync
uv run vg demo --tiny          # honest PASS, four tamper FAILs with reason codes, cost summary
uv run pytest -q               # canon vs transformers, Merkle, keygen, protocol, adversarial suite
```

By hand, over HTTP:

```bash
uv run vg tiny --out ./tiny-gemma4
uv run vg keygen --model ./tiny-gemma4 --out ./keys/tiny                 # key.npz (secret) + public.json
uv run vg serve --model ./tiny-gemma4 --public ./keys/tiny/public.json   # the server never reads key.npz
uv run vg chat --model ./tiny-gemma4 --prompt "Hello" --max-new-tokens 16  # receipt.json + request.json
uv run vg audit                                        # random per-position audits, every token covered
uv run vg verify --key ./keys/tiny/key.npz --public ./keys/tiny/public.json --request request.json --deep
uv run vg serve ... --tamper weights|identity|sampling|softcap           # demo only, logs a loud warning
```

## What the demo shows

On Gemma 4 12B-it (A100-40GB, Modal; the routine audit draws 10 of 48 layers per full audit):

```
  [ok ] honest    routine:10 PASS                  (expected PASS)
  [ok ] honest    full       PASS                  (expected PASS)
  [ok ] weights   full       FAIL FREIVALDS_WDOWN  (expected FREIVALDS_WDOWN)  layer 24  deviation 0.147 > tolerance 0.0663
  [ok ] identity  full       FAIL FREIVALDS_WQ     (expected FREIVALDS_WQ)  layer 0  deviation 3.49e+03 > tolerance 23.4
  [ok ] sampling  full       FAIL DECODE_SAMPLING  (expected DECODE_SAMPLING)
  [ok ] softcap   full       FAIL DECODE_SOFTCAP   (expected DECODE_SOFTCAP)
```

On the tiny CPU model:

```
  [ok ] honest    routine:3  PASS                  (expected PASS)
  [ok ] honest    full       PASS                  (expected PASS)
  [ok ] weights   full       FAIL FREIVALDS_WDOWN  (expected FREIVALDS_WDOWN)  layer 3  deviation 0.0188 > tolerance 0.0073
  [ok ] identity  full       FAIL FREIVALDS_WQ     (expected FREIVALDS_WQ)  layer 0  deviation 4.81 > tolerance 0.0444
  [ok ] sampling  full       FAIL DECODE_SAMPLING  (expected DECODE_SAMPLING)
  [ok ] softcap   full       FAIL DECODE_SOFTCAP   (expected DECODE_SOFTCAP)
```

Each scenario starts a fresh provider process that receives only `public.json` and a state
directory. The demo process holds the key and plays client, auditor and verifier. It sends the
prompt with a fresh nonce, checks the receipt against its own templated prompt and sampling
policy, and pins the prover's key. Then it draws per-position audits: full audits of three random
generated positions, each with its own routine (or full) layer set, plus a decode audit with one
random layer for every other generated token. It receives the opening over HTTP and verifies it. The tamper modes are:

- `weights`: a rank-8 delta of 3 % merged into one `down_proj`;
- `identity`: a different checkpoint, here two layers swapped, served behind the declared root;
- `sampling`: temperature changed while the manifest says 1.0;
- `softcap`: the logit soft-cap skipped.

## Claim boundary

This uses CommitLLM's vocabulary. Verified means checked independently from committed data and
verifier-secret randomness. Audited means checked, but not a verification of the computation. Open
means committed, not checked.

| Component | Status | How |
|---|---|---|
| Embedding row and scale | verified, exact | Merkle proof to the embedding root; `r_in[0] == bf16(row * 62.0)` bit for bit |
| Linear shell Wq Wk Wv Wo Wgate Wup Wdown | verified, tolerance-bounded | Freivalds with 16 secret vectors per family; bf16 bound printed with the deviation |
| LM head (logits computed in f32) | verified, tolerance-bounded | Freivalds binding of the logits to the final hidden state, f32 bound |
| Norms: input, post-attention, pre-FFN, post-FFN, q, k, v (no weight), final | verified, tolerance-bounded | float64 replay, elementwise `2^-7` relative |
| GELU-tanh gate | verified, tolerance-bounded | float64 replay, elementwise `2^-6` relative |
| Residual chain with `layer_scalar` | verified, exact | bf16 round-to-nearest-even replay of both adds and the scalar |
| Shared K and V on global layers | verified, 1 ulp | `v_n == bf16(rmsnorm(k before k_norm))` |
| Soft-cap, at every generated token | verified, exact | canonical f32 soft-cap, bit for bit |
| Sampled token, at every generated token | verified, exact | shared sampler; the uniform comes from a seed fixed by the client's nonce |
| Bindings: prompt, client request (policy, max tokens, nonce), manifest, seed, token count, stop rule, IO chain, signature | verified, exact | hashes, Ed25519 with a pinned prover key, Merkle |
| K/V provenance | audited | every attended K/V row Merkle-bound to the trace root |
| Attention output at challenged positions | audited | single-query float64 replay of RoPE (bf16-faithful), scores, mask, softmax, AV |
| Wiring: layer types, sliding window, head dims, KV heads, partial RoPE, QK-norm, no Wv on global layers | audited | manifest hashes and opened shapes |
| Layers the auditor did not draw at a position | open (spot check) | committed. Every generated token gets its own random layer audit, so a residual stream forged at one layer boundary at every token escapes with `prod (1 - layers_a/L)`, printed in each verdict: 6e-4 on tiny with 32 tokens; about `0.5 × (47/48)^n` on 12B, i.e. 0.30 at 24 tokens and 0.03 at 128 |
| Consistent fake attention output `a` at a layer the attention replay does not cover | open | the residual hole CommitLLM documents. Only audits with the attention replay catch it; the verdict prints that escape probability separately (0.125 on tiny, about 0.5 on 12B with 3 full audits). `vg audit --decode-attention` extends the replay to every decode audit, at the cost of opening the K/V rows |
| Shell deviations below the bf16 bound | open in Tier 1 | up to about 1.2 % of a projection output's L2 norm, possibly concentrated on a few elements; the exact INT8 path (Tier 2) closes it |
| Sampling randomness when the client sends no nonce | open | the prover's own seed allows best-of-N grinding; `vg chat` always sends a nonce |

Tolerance-bounded checks are not exact, and attention replay is not described as verification.
Every verdict prints, for each check, the worst observed deviation, the bound and the margin.

### Tolerances, and why they are what they are

| Check | Bound | Honest worst, tiny CPU | Honest worst, 12B on A100 (1,500+ checks) |
|---|---|---|---|
| Freivalds (every shell family) | `RMS_j(r_j·y - v_j·x) <= (3·2^-8 + 8·sqrt(K)·2^-24)·‖y‖₂` | 20 to 24 % | 28 % (p99 23 %) |
| LM-head binding (f32 logits) | same statistic with `3·2^-24` instead of `3·2^-8` | 1.3 % | 0.4 % |
| Norms | `|y - ref| <= 2^-7·|ref| + 2^-126` elementwise | 50 % | 49.8 % |
| GELU gate | `|h - ref| <= 2^-6·|ref| + 2^-20·|g·u| + 2^-126` | 45 % | 49.3 % |
| KV_SHARED | 1 bf16 ulp | 0 ulps | 0 ulps |
| Attention replay (audited), with the declared kernel's rounding | per head `‖a - ref‖₂ / ‖ref‖₂ <= 2^-5` | 12 % | 13 % (p99 11 %) |

The draft spec's Freivalds bound (`2^-7·‖y‖₁` with one vector) could not detect its own 3 % weights
tamper on any real layer, because a ±1 projection of the change grows with `‖Δy‖₂` while that bound
grows with `sqrt(m)·‖y‖₂`. The implemented statistic averages 16 secret projections. Its bound holds
with probability above 1 - 10^-20 over the secret vectors for an honest prover, and it rejects any
change larger than about 1.2 % of an output's L2 norm (TECH_SPEC section 11). The attention replay
follows the rounding of the declared `attn_implementation`: flash-style bf16 probabilities for
`sdpa`, bf16 scores and weights for `eager`. An exact f64 replay deviated up to 0.029 on the GPU
and produced one honest false positive in the first real-model demo; with the matching model the
worst of 1,500+ GPU replays is 0.0075, about 4x inside the bound (DECISIONS 44).

Detection power on the real model is thinner than on tiny. The spec's 3 % rank-8 `down_proj`
tamper on Gemma 4 12B was caught in all three demo runs, at 1.07x, 1.7x and 2.2x the bound
depending on the audited position. Changes at that size sit near the Tier 1 bf16 threshold; the
exact Tier 2 path is what makes such small changes reliably detectable.

## Metrics

`vg bench` and `vg demo` print these. The tiny CPU numbers come from an Apple M4 Max with torch
2.14.1 and transformers 5.18.0: 86-token prompt, 32 generated tokens, 3 challenged positions. On a
model this small the timing overheads are noisy between runs.

| Metric | tiny, CPU | gemma-4-12B-it, A100-40GB |
|---|---|---|
| Decode tokens/s, capture off / on | about 340, overhead -6 to 15 % across runs | 12.2 / 9.3 (24 % overhead; 19 % in the demo run) |
| Serve tokens/s incl. commitment, off / on | overhead 17 to 29 % | 12.1 / 6.5 (47 %: hashing about 10 MiB per position, 1.3 s for 56 positions) |
| Retained state per position | 16.3 KiB | 10.2 MiB |
| Opening per full audit, routine / all layers | 76 KiB / 167 KiB (3 / 6 layers) | 4.4 MiB / 16.9 MiB (10 / 48 layers) |
| Opening per audit without attention, routine / all / one layer | 42 KiB / 51 KiB / 34 KiB | 3.0 MiB / 10.1 MiB / 1.2 MiB |
| Verifier per full audit, routine / all layers; per one-layer audit | 4.1 ms / 8.3 ms; 1.6 ms | 216 ms / 894 ms; 39 ms (CPU) |
| Default challenge on 32 tokens (3 full + 29 decode audits) | 638 KiB, 45 ms, forged-boundary escape 6.3e-4 | 51.9 MiB, 1.77 s, forged-boundary escape 0.27 |
| Keygen time, key size | 0.01 s, 489 KiB (k = 16) | 98 s, 268.6 MiB (k = 16) |

The GPU throughput is Hugging Face eager decoding at batch size 1 with no CUDA graphs or
`torch.compile`, plus the CPU soft-cap and sampler (2.3 ms per token at a 262k vocabulary). Serving
overhead on the real model is dominated by hashing the Tier 1 bf16 retained state (about 10 MiB
per position). Tier 2's compact INT8 retained state is what brings that down.

## Real model on GPU (Modal, vast.ai as backup)

```bash
uv sync --extra modal
uv run modal run infra/modal_app.py::download --model google/gemma-4-12B-it
uv run modal run --detach infra/modal_app.py::keygen --model google/gemma-4-12B-it
uv run modal run infra/modal_app.py::demo --model google/gemma-4-12B-it
uv run modal run infra/modal_app.py::bench --model google/gemma-4-12B-it
VG_GPU=H100 uv run modal run infra/modal_app.py::demo --model google/gemma-4-31B-it
```

The key never enters the serving function: the `vg-keys` volume is mounted only by `keygen`,
`demo` and `bench`, and the provider runs as a subprocess that is given only `/state` paths. The
vast.ai path runs the same `uv run vg ...` commands; see [infra/vast/README.md](infra/vast/README.md).

## How it works

1. **Keygen** (`vg keygen`) streams the public checkpoint once and produces:
   - the weights root (every tensor) and the embedding root (every vocabulary row);
   - for each layer and each matrix family of its layer type, 16 secret ±1 vectors `r` and
     `v = rᵀW` in float64 (global layers have no Wv);
   - the same for the tied LM head;
   - the norm weights, the `layer_scalar` values and the canonical config hash.
2. **Commit** (`vg serve`, `/chat`): the provider runs its own decode loop with a KV cache. Forward
   hooks capture 18 tensors per layer and position, staying on the device in bf16 until commit. The
   engine computes the logits in f32 from the final hidden state. The soft-cap and the sampler run
   on CPU, with randomness fixed by the client's nonce. The provider then:
   - hashes every tensor into layer leaves, position leaves and a Merkle trace root;
   - chains the tokens and logits into an IO chain;
   - signs a receipt binding the model, the manifest, the prompt, the seed commitment and the counts;
   - keeps the retained state on disk for an audit window.
3. **Audit** (`vg audit`, `/audit`): the auditor draws random positions and layers after it has the
   receipt, plus decode checks on every generated token. The provider opens exactly what the checks
   need: challenged layers, the next layer's input, the embedding row, the pre-cap logits, every
   attended K/V row, and Merkle paths.
4. **Verify** (`vg verify`): numpy only, no torch, no weights, given the auditor's own challenge and
   the client's request. It checks:
   - bindings (including the request), then wiring, then Merkle membership;
   - the embedding, Freivalds for each family, and the bridge replay (four norms, QK-norm, V-norm,
     GELU gate, exact residuals with `layer_scalar`);
   - shared K/V and the attention replay (audited);
   - the LM-head binding, the exact soft-cap and the exact sampled token.

   The first failing check gives the reason code.

Gemma 4 specifics handled in the profile include:

- the 5:1 sliding/global pattern;
- per-layer-type head dims and KV heads (256/8 vs 512/1 on 12B);
- shared K/V and `v_norm` on global layers;
- proportional partial RoPE (25 %) with theta 1e6;
- attention scale 1.0;
- bf16-rounded embedding scale;
- tied head with soft-cap 30;
- non-trivial `layer_scalar`.

Unsupported variants fail closed at keygen and at serve: MoE, per-layer embeddings, KV-shared
layers, double-wide MLP, bidirectional text attention, untied heads, unknown layer types.

## Security review

After the first version, an independent adversarial review of the verifier found six ways to
obtain a PASS outside the claim boundary, each confirmed with a working PoC:

- an infinite logit passed the LM-head bound (`inf <= inf`) and forced any output;
- a layer group could carry the committed leaf beside fabricated tensors;
- the default challenge always opened the first, middle and last token;
- the prompt binding could be skipped by not opening the prompt;
- `vg verify` trusted the prover's echo of the challenge;
- with bf16 logits, the L2 LM-head bound let one logit move enough to choose the token.

All six are fixed. Each PoC is now a test in `tests/test_adversarial_review.py`, and the reasoning
is recorded in [docs/DECISIONS.md](docs/DECISIONS.md) (26 to 36). The review also led to the
client nonce, which stops seed grinding, and to decode checks on every generated token.

A second pass over the fixes found three more holes, also fixed and pinned by tests (DECISIONS
37 to 40):

- A residual stream forged at one layer boundary at every token was caught only if the single
  shared layer set held the producing layer. Audits are now per position, and every token gets its
  own random layer.
- Prompt tokens at unopened positions were not bound to the revealed prompt. Every position's input
  token is now bound through a two-level leaf.
- The client's prompt binding was optional. The client now always templates its own prompt.

A third, narrower pass found two medium issues, now fixed (DECISIONS 42 and 43):

- The single escape probability overstated how well decode audits cover the attention path. The
  verdict now reports a separate number for a fake attention output.
- Three client fields were recorded but not enforced: the attention implementation and the
  tokenizer and template hashes. `check_expected` now checks them.

## Adversarial suite

`tests/test_adversarial_tiny.py` and `tests/test_adversarial_review.py` contain about 177
scenarios, each rejected for the expected reason code:

- all tamper modes, including a weights delta on every layer and on every family;
- bit flips in every captured tensor of both layer types, in the logits and in the embedding row;
- swapped layers and positions, wrong Merkle siblings, truncated paths, a wrong seed;
- prompt and transcript edits, cross-request splices, withheld tensors, removed `v_norm`, a `v_proj`
  on a global layer, wrong head shapes;
- receipt edits, both after signing and re-signed by the prover: manifest fields, counts and roots;
- a prover that commits to a consistently modified trace;
- a prover serving a consistently modified model: SiLU instead of GELU, Llama-style `(1 + w)` norm,
  `layer_scalar` dropped, wrong embedding scale, untied head, sliding window 16, `1/sqrt(d)` scaling,
  full RoPE on global layers, skipped `q_norm` or `v_norm`;
- the review regressions: non-finite logits, leaf plus tensors, forged tokens outside the challenged
  positions, prompt and policy substitution, prover-chosen challenges, single-logit boosts, and
  seeds that ignore or grind the client nonce.

It mirrors CommitLLM's adversarial scenario list.

## Layout

```
src/vgemma/   profile.py model.py canon.py merkle.py protocol.py keygen.py tiny.py tokenizer.py
              auditor.py demo.py display.py cli.py
  prover/     hooks.py sampler.py engine.py store.py server.py tamper.py
  verifier/   freivalds.py bridge.py attention.py decode.py bindings.py verify.py key.py codes.py report.py
tests/        test_canon.py test_merkle.py test_keygen.py test_protocol_tiny.py test_adversarial_tiny.py
              test_adversarial_review.py test_gpu_smoke.py
infra/        modal_app.py vast/setup.sh vast/README.md
docs/         DECISIONS.md SCHEMAS.md
```

## Licence and attribution

MIT ([LICENSE](LICENSE)). The protocol design, vocabulary and adversarial scenario list come from
CommitLLM by LambdaClass (MIT); no CommitLLM code is copied. Gemma 4 is by Google DeepMind (Apache
2.0); no weights are distributed here. See [NOTICE](NOTICE).
