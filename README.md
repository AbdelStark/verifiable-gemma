<div align="center">

# verifiable-gemma

**Signed, auditable receipts for every Gemma 4 response.**

The provider commits to the execution trace while it serves. An auditor holding a small secret
key later challenges random positions and layers, and checks on a CPU, without the model weights,
that the declared checkpoint, configuration and sampling policy produced the answer.

[![Tests](https://img.shields.io/github/actions/workflow/status/AbdelStark/verifiable-gemma/ci.yml?branch=main&style=for-the-badge&logo=githubactions&logoColor=white&label=tests)](https://github.com/AbdelStark/verifiable-gemma/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow?style=for-the-badge)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](.python-version)
[![uv](https://img.shields.io/badge/packaged%20with-uv-DE5FE9?style=for-the-badge&logo=uv&logoColor=white)](https://docs.astral.sh/uv/)

[![Gemma 4](https://img.shields.io/badge/model-Gemma%204-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://huggingface.co/google/gemma-4-12B-it)
[![Protocol: CommitLLM](https://img.shields.io/badge/protocol-CommitLLM-0B0B0B?style=for-the-badge)](https://github.com/lambdaclass/CommitLLM)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/🤗%20transformers-5.18-FFD21E?style=for-the-badge)](https://github.com/huggingface/transformers)
[![Verified on A100](https://img.shields.io/badge/demo-passing%20on%20A100-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](#results)

[Why](#why) · [Quickstart](#quickstart) · [How it works](#how-it-works) · [Claim boundary](#what-is-verified-what-is-audited-what-is-open) · [Results](#results) · [Security review](#security-review) · [Docs](#documentation)

</div>

---

## Why

When a model is served behind an API, only the operator knows which weights answered, at which
temperature, with which configuration. Fingerprinting gives signals, not evidence. Zero-knowledge
proofs of inference cost orders of magnitude more than the inference. Trusted hardware moves the
trust instead of removing it.

Open weights change the economics. Anyone can build a verification key from a public checkpoint,
so the provider only has to **commit** to what it computed and **open** what is challenged.
[CommitLLM](https://github.com/lambdaclass/CommitLLM) showed this for Llama and Qwen.
`verifiable-gemma` brings the protocol to **Gemma 4**, whose architecture makes that real work:

- sliding and global attention layers;
- QK-norm, and K and V shared on global layers;
- proportional partial RoPE;
- GELU-tanh gating;
- a logit soft-cap and tied embeddings;
- a learned `layer_scalar` on every layer.

What you get:

- 🧾 **A signed receipt with every response**, binding the weights, configuration, manifest,
  prompt, sampling seed, token count and the Merkle root of the execution trace.
- 🔍 **Audits on demand.** The auditor draws random positions and layers after the fact. The
  provider opens only those, plus Merkle proofs.
- 🖥️ **CPU-only verification.** The verifier needs numpy and a 269 MiB key, no torch and no
  weights. A default audit of 32 tokens on Gemma 4 12B verifies in about 1.8 s.
- 🚨 **Tampering caught, with a reason code.** A merged weight delta, a swapped checkpoint, a
  changed temperature or a removed soft-cap each fail with a named reason.
- 📏 **An honest claim boundary.** Every check is labelled exact, tolerance-bounded (with the bound
  and observed deviation printed) or audited. What stays open is listed below.

## At a glance

On `google/gemma-4-12B-it`, one A100-40GB, launched from [`infra/modal_app.py`](infra/modal_app.py),
each scenario gets a fresh provider process:

```text
  [ok ] honest    routine:10 PASS                  (expected PASS)
  [ok ] honest    full       PASS                  (expected PASS)
  [ok ] weights   full       FAIL FREIVALDS_WDOWN  (expected FREIVALDS_WDOWN)  layer 24  deviation 0.147 > tolerance 0.0663
  [ok ] identity  full       FAIL FREIVALDS_WQ     (expected FREIVALDS_WQ)     layer 0   deviation 3.49e+03 > tolerance 23.4
  [ok ] sampling  full       FAIL DECODE_SAMPLING  (expected DECODE_SAMPLING)
  [ok ] softcap   full       FAIL DECODE_SOFTCAP   (expected DECODE_SOFTCAP)

DEMO PASSED: honest PASS, every tamper rejected with the expected reason
```

| Tamper mode | What the cheating provider does | Caught by |
|---|---|---|
| `weights` | merges a rank-8 delta (3 % of the norm) into one `down_proj` | `FREIVALDS_WDOWN` |
| `identity` | serves a different checkpoint (two layers swapped) behind the declared root | `FREIVALDS_WQ` |
| `sampling` | samples at temperature 0.25 while the manifest says 1.0 | `DECODE_SAMPLING` |
| `softcap` | skips the final logit soft-cap | `DECODE_SOFTCAP` |

## Quickstart

Everything runs on a laptop CPU against a tiny random Gemma 4. It keeps every architectural
feature: the 5:1 sliding/global pattern, shared K/V, partial RoPE, QK-norm, the soft-cap, tied
embeddings and non-trivial `layer_scalar`.

```bash
git clone https://github.com/AbdelStark/verifiable-gemma && cd verifiable-gemma
uv sync
uv run vg demo --tiny        # honest PASS, four tamper FAILs, cost summary  (~15 s)
uv run pytest -q             # 258 tests: canon vs transformers, protocol, adversarial suite
```

Or play every role yourself over HTTP:

```bash
uv run vg tiny   --out ./tiny-gemma4
uv run vg keygen --model ./tiny-gemma4 --out ./keys/tiny               # key.npz (secret) + public.json
uv run vg serve  --model ./tiny-gemma4 --public ./keys/tiny/public.json # the provider; never reads key.npz

# in a second terminal: client, auditor, verifier
uv run vg chat   --model ./tiny-gemma4 --prompt "Hello" --max-new-tokens 16   # receipt.json + request.json
uv run vg audit                                                             # challenge.json + opening.bin
uv run vg verify --key ./keys/tiny/key.npz --public ./keys/tiny/public.json --request request.json --deep
```

To watch a provider get caught, restart `vg serve` with `--tamper weights|identity|sampling|softcap`
(demo only, with a loud warning).

## How it works

### The three parties

```text
                         ┌─────────────────────────────────────┐
                         │      public Gemma 4 checkpoint      │
                         │  safetensors · config · tokenizer   │
                         └──────────────────┬──────────────────┘
                                            │  vg keygen  (once, offline: 98 s on 12B)
                        ┌───────────────────┴──────────────────────┐
                        ▼                                          ▼
          ┌───────────────────────────┐            ┌───────────────────────────────┐
          │ public.json               │            │ key.npz              (SECRET) │
          │  weights root             │            │  16 random ±1 vectors per W   │
          │  embedding root           │            │  v = rᵀW in float64           │
          │  config + tokenizer hash  │            │  norm weights, layer scalars  │
          └─────────────┬─────────────┘            └───────────────┬───────────────┘
                        │ handed to the provider                   │ stays with the auditor
                        ▼                                          ▼
 ┌────────────────────────────────────────┐    ┌────────────────────────────────────────┐
 │ PROVIDER          GPU · vg serve       │    │ CLIENT / AUDITOR    CPU · numpy only   │
 │                                        │    │                                        │
 │ transformers + capture hooks           │◀───│ 1  /chat   prompt, policy, nonce       │
 │ own decode loop, f32 LM head           │───▶│ 2          text + signed receipt       │
 │ canonical soft-cap + sampler on CPU    │    │                                        │
 │ Merkle-commits 18 tensors per layer    │◀───│ 3  /audit  random per-position audits  │
 │ keeps retained state for a TTL         │───▶│ 4          opening: tensors + proofs   │
 │                                        │    │                                        │
 │ reads public.json, never key.npz       │    │ vg verify → PASS / FAIL + reason code  │
 └────────────────────────────────────────┘    └────────────────────────────────────────┘
```

The provider never sees the Freivalds vectors. On Modal this is enforced by the infrastructure:
the `vg-keys` volume is never mounted into the serving function.

### One request, end to end

```text
  client / auditor                                   provider (GPU)
        │                                                  │
        │  POST /chat   prompt · policy · nonce            │
        │─────────────────────────────────────────────────▶│  forward pass, hooks capture every layer
        │                                                  │  logits in f32 → soft-cap → sample (CPU)
        │                                                  │  randomness u_t = H(H(nonce) ‖ t)
        │                                                  │  commit: tensors → leaves → trace root
        │  text + receipt (≈ 2 KB, Ed25519-signed)         │          tokens + logits → IO chain
        │◀─────────────────────────────────────────────────│  retain trace (10 MiB per position)
        │                                                  │
        │  … any time within the audit window …            │
        │                                                  │
        │  POST /audit  [{pos, layers, attention}, …]      │
        │─────────────────────────────────────────────────▶│  open exactly what those checks need
        │  opening: tensors + Merkle proofs                │
        │◀─────────────────────────────────────────────────│
        ▼
  vg verify  (CPU, no weights)
        bindings → wiring → Merkle → embedding → shell → bridges → attention → decode
        → PASS, or FAIL with the first reason code, plus coverage and bounds
```

The auditor draws the challenge with its own randomness after it holds the receipt. By default
that is:

- three random generated positions with a **full audit** (an independent routine subset of 10
  layers each on 12B, plus the attention replay);
- every other generated token with a **decode audit**: one random layer audited in full, plus the
  LM-head binding, the soft-cap and the sampled token;
- every remaining position opened just enough to bind its input token.

### What gets checked inside one Gemma 4 decoder layer

```text
  token ─▶ embed_tokens[token] × 62.0 ─▶ r_in                           [=] Merkle row + exact bf16

  ┌─ decoder layer ℓ   48 on 12B · 5 sliding (window 1024) : 1 global ──────────────────────────┐
  │                                                                                             │
  │  x_attn = input_layernorm(r_in)                                             [B]             │
  │  q, k, v = Wq·x_attn, Wk·x_attn, Wv·x_attn       (global: no Wv, V from K)  [F]             │
  │  q_n, k_n = q_norm(q), k_norm(k)                 (QK-norm, per head)        [B]             │
  │  v_n = v_norm(v)    sliding                      (RMSNorm, no weight)       [B]             │
  │  v_n = v_norm(k)    global                       (shared K and V)           [≤1 ulp]        │
  │  a = softmax( RoPE(q_n) · RoPE(k_n)ᵀ ) · v_n     (scale 1, causal window)   [A] replay+KV   │
  │  o = Wo·a                                                                   [F]             │
  │  r_mid = r_in + post_attention_layernorm(o)                                 [B] + [=]       │
  │  x_ffn = pre_feedforward_layernorm(r_mid)                                   [B]             │
  │  g, u = Wgate·x_ffn, Wup·x_ffn                                              [F]             │
  │  h = gelu_tanh(g) ⊙ u                                                       [B]             │
  │  d = Wdown·h                                                                [F]             │
  │  r_out = (r_mid + post_feedforward_layernorm(d)) × layer_scalar             [B] + [=]       │
  │                                                                                             │
  └─────────────────────────────────────────────────────────────────────────────────────────────┘
       │ r_final
       ▼
  h = final_norm(r_final)                                                       [B]
  z = E·h            f32, tied embedding                                        [F] f32 bound
  z' = 30·tanh(z / 30)                                                          [=] bit-exact
  token = sample(z', T, top_k, top_p, u_t)                                      [=] bit-exact

  [F] Freivalds: 16 secret vectors, bf16 bound      [B] float64 replay within a stated bound
  [=] exact, bit for bit                             [A] audited: replayed, not verified
```

### How the trace is committed

```text
  receipt ── signed ──▶ trace_root ─┬─ position leaf p = H(p ‖ input token ‖ body)
                                    │      body = H(layer leaves 0..48 ‖ H(logits) ‖ token ‖ H(witness))
                                    │      layer leaf ℓ = H(ℓ ‖ p ‖ H(r_in) ‖ H(x_attn) ‖ … ‖ H(d_n))
                                    │
            io_chain_head ──────────┼─ c_t = H(c_(t-1) ‖ token_t ‖ H(logits_t)),  c_0 = H(prompt)
            seed_commitment ────────┼─ H(seed ‖ request id),  seed = H(client nonce)
            weights_root ───────────┴─ Merkle root over every checkpoint tensor (computed at keygen)
```

Every hash is domain-separated SHA-256. Opened tensors are hashed from their canonical bytes and
must reproduce the committed leaves. A layer group is opened either as tensors or as a leaf, never
both. Every position's input token is bound to the revealed prompt or to the generated transcript.
Formats are in [docs/SCHEMAS.md](docs/SCHEMAS.md).

## What is verified, what is audited, what is open

This uses CommitLLM's vocabulary:

- **Verified:** checked independently from committed data and verifier-secret randomness.
- **Audited:** checked, but not a verification of the computation.
- **Open:** committed, not checked.

| Component | Status | How |
|---|---|---|
| Embedding row and scale | ✅ verified, exact | Merkle proof to the embedding root; `r_in[0] == bf16(row × 62.0)` bit for bit |
| Linear shell Wq Wk Wv Wo Wgate Wup Wdown | ✅ verified, tolerance-bounded | Freivalds with 16 secret vectors per family; bf16 bound printed with the deviation |
| LM head (logits computed in f32) | ✅ verified, tolerance-bounded | Freivalds binding of the logits to the final hidden state, f32 bound |
| Norms: input, post-attention, pre-FFN, post-FFN, q, k, v, final | ✅ verified, tolerance-bounded | float64 replay, elementwise `2^-7` relative |
| GELU-tanh gate | ✅ verified, tolerance-bounded | float64 replay, elementwise `2^-6` relative |
| Residual chain with `layer_scalar` | ✅ verified, exact | bf16 round-to-nearest-even replay of both adds and the scalar |
| Shared K and V on global layers | ✅ verified, 1 ulp | `v_n == bf16(rmsnorm(k before k_norm))` |
| Soft-cap, at every generated token | ✅ verified, exact | canonical f32 soft-cap, bit for bit |
| Sampled token, at every generated token | ✅ verified, exact | shared sampler; randomness fixed by the client's nonce |
| Bindings: prompt, client request, manifest, seed, token count, stop rule, IO chain, signature | ✅ verified, exact | hashes, Ed25519 with a pinned prover key, Merkle |
| K/V provenance | 🔎 audited | every attended K/V row Merkle-bound to the trace root |
| Attention output at audited positions | 🔎 audited | single-query replay with the declared kernel's rounding (`sdpa` / `eager`) |
| Wiring: layer types, window, head dims, KV heads, partial RoPE, QK-norm, no Wv on global layers | 🔎 audited | manifest hashes and opened shapes |
| Layers the auditor did not draw at a position | ⚪ open, spot check | a residual stream forged at one layer boundary at every token escapes with `∏ (1 − layers_a / L)`, printed in each verdict: 0.27 for a 32-token answer on 12B, 6e-4 on tiny |
| Consistent fake attention output at a layer without the attention replay | ⚪ open | the residual hole CommitLLM documents; the verdict prints its escape probability separately (about 0.5 on 12B); `vg audit --decode-attention` extends the replay |
| Shell deviations below the bf16 bound | ⚪ open in Tier 1 | up to about 1.2 % of a projection output's L2 norm, possibly concentrated; the exact INT8 path (Tier 2) closes it |
| Sampling randomness without a client nonce | ⚪ open | the prover's own seed allows best-of-N grinding; `vg chat` always sends a nonce |

Tolerance-bounded checks are not exact, and the attention replay is not described as
verification. Every verdict prints, for each check, the worst observed deviation, the bound and
the margin.

## Results

`vg bench` and `vg demo` print these numbers. Tiny ran on an Apple M4 Max (86-token prompt, 32
generated tokens). The 12B runs used one A100-40GB on Modal (25-token prompt, 32 generated
tokens).

| Metric | tiny, CPU | gemma-4-12B-it, A100-40GB |
|---|---|---|
| Decode tokens/s, capture off / on | ≈ 340, overhead −6 to 15 % | 12.2 / 9.3 (24 % overhead) |
| Serve tokens/s including commitment, off / on | overhead 17 to 29 % | 12.1 / 6.5 (47 %) |
| Retained state per position | 16.3 KiB | 10.2 MiB |
| Opening per full audit, routine / all layers | 76 KiB / 167 KiB | 4.4 MiB / 16.9 MiB |
| Opening per audit without attention, routine / all / one layer | 42 / 51 / 34 KiB | 3.0 / 10.1 / 1.2 MiB |
| Verifier per full audit, routine / all layers; per one-layer audit | 4.1 / 8.3 ms; 1.6 ms | 216 / 894 ms; 39 ms |
| **Default challenge on 32 tokens** (3 full + 29 decode audits) | 638 KiB, 45 ms | **51.9 MiB, 1.77 s** |
| Keygen time, key size | 0.01 s, 489 KiB | 98 s, 268.6 MiB |

The GPU throughput is Hugging Face eager decoding at batch size 1, with no CUDA graphs and no
`torch.compile`. Serving overhead on 12B is dominated by hashing about 10 MiB of bf16 retained
state per position; the compact INT8 retained state of Tier 2 is what brings it down.

### Tolerances, measured

Each bound is derived from the arithmetic and printed in every verdict. The measured columns are
honest worst cases as a share of the bound; on 12B that covers more than 1,500 checks of each kind.

| Check | Bound | tiny CPU | 12B A100 |
|---|---|---|---|
| Freivalds (every shell family) | `RMS_j(r_j·y − v_j·x) ≤ (3·2^-8 + 8·√K·2^-24)·‖y‖₂` | 20–24 % | 28 % (p99 23 %) |
| LM-head binding, f32 logits | same statistic with `3·2^-24` | 1.3 % | 0.4 % |
| Norms | `\|y − ref\| ≤ 2^-7·\|ref\| + 2^-126` elementwise | 50 % | 49.8 % |
| GELU gate | `\|h − ref\| ≤ 2^-6·\|ref\| + 2^-20·\|g·u\| + 2^-126` | 45 % | 49.3 % |
| Shared K/V | 1 bf16 ulp | 0 ulps | 0 ulps |
| Attention replay (audited) | per head `‖a − ref‖₂ / ‖ref‖₂ ≤ 2^-5` | 12 % | 13 % (p99 11 %) |

- **Freivalds.** The draft spec used one random vector with an L1 bound, which cannot detect its
  own 3 % weights tamper on any real layer. Sixteen secret projections with an L2 bound hold for an
  honest prover with probability above 1 − 10⁻²⁰. They reject changes larger than about 1.2 % of
  an output's L2 norm.
- **Attention.** The replay follows the rounding of the declared kernel. An exact float64 replay
  produced an honest false positive on the GPU. See [DECISIONS 44](docs/DECISIONS.md).
- **Detection margin on 12B.** The 3 % rank-8 `down_proj` tamper was caught in all three demo runs,
  at 1.07×, 1.7× and 2.2× the bound depending on the audited position. Changes of that size sit
  near the Tier 1 bf16 threshold.

## Running on real Gemma 4

GPU work is driven from code on [Modal](https://modal.com). The image is built from this
repository's `uv.lock`, and three volumes hold the weights cache, the state and the keys.

```bash
uv sync --extra modal
uv run modal run infra/modal_app.py::download --model google/gemma-4-12B-it   # 24 GB, once
uv run modal run --detach infra/modal_app.py::keygen --model google/gemma-4-12B-it
uv run modal run infra/modal_app.py::demo  --model google/gemma-4-12B-it      # A100-40GB
uv run modal run infra/modal_app.py::bench --model google/gemma-4-12B-it      # metrics + calibration
VG_GPU=H100 uv run modal run infra/modal_app.py::demo --model google/gemma-4-31B-it
```

vast.ai is the backup launcher and runs the same `uv run vg …` commands over SSH; see
[infra/vast/README.md](infra/vast/README.md). The `serve` web endpoint is unauthenticated, so add
proxy auth before any `modal deploy`.

Supported checkpoints are the dense Gemma 4 text decoders, `gemma4_text` and
`gemma4_unified_text`. The 12B has been run end to end. The 31B uses the same decoder and has
not been run yet. Every other variant **fails closed** at keygen and at
serve: MoE, per-layer embeddings (E2B/E4B), KV-shared layers, double-wide MLP, bidirectional text
attention, untied heads, unknown layer types.

## Security review

The verifier went through three adversarial review passes. Each confirmed hole became a regression
test in [`tests/test_adversarial_review.py`](tests/test_adversarial_review.py) before being fixed.

| Pass | Found | Fix |
|---|---|---|
| 1 | an infinite logit passed `inf ≤ inf` and forced any output | non-finite values fail closed |
| 1 | a committed leaf next to fabricated tensors | a group carries a leaf **or** tensors |
| 1 | predictable challenge positions | the auditor draws positions at random |
| 1 | the prompt binding could be skipped; the challenge echo was trusted | prompt always revealed; `verify` takes the auditor's own challenge and the client's request |
| 1 | with bf16 logits, one logit could move enough to pick the token | the LM head is computed in f32 (bound about 400× tighter on 12B) |
| 2 | one forged residual boundary steered every token past routine audits | per-position audits, a random layer at every token, escape probability in the verdict |
| 2 | prompt tokens at unopened positions were unbound | two-level position leaf; every input token bound |
| 3 | escape number overstated attention coverage; three client fields unchecked | separate attention escape number; all client fields enforced |

The full adversarial suite has about 180 scenarios, mirroring CommitLLM's scenario list and
extending it to Gemma 4. Each must be rejected with the expected reason code. It covers:

- every tamper mode, and a weights delta on every layer and family;
- bit flips in every captured tensor;
- swapped layers and positions, wrong Merkle siblings, wrong seeds;
- cross-request splices and withheld tensors;
- receipt edits, both after signing and re-signed by the prover;
- consistent malicious recommitments;
- consistently modified models: SiLU instead of GELU, a Llama-style `(1 + w)` norm, a dropped
  `layer_scalar`, an untied head, a wrong sliding window, `1/√d` scaling, full RoPE on global
  layers, a skipped `q_norm` or `v_norm`.

## CLI

| Command | Role | What it does |
|---|---|---|
| `vg tiny` | dev | build the tiny random Gemma 4 checkpoint |
| `vg keygen` | auditor | secret `key.npz` + `public.json` from a public checkpoint |
| `vg serve` | provider | HTTP server with capture hooks: `/chat`, `/audit`, `/health` (`--tamper` for demos) |
| `vg chat` | client | send a prompt with a fresh nonce; save `receipt.json` and `request.json` |
| `vg audit` | auditor | draw a random challenge; save `challenge.json` and `opening.bin` |
| `vg verify` | auditor | CPU verdict: PASS / FAIL, reason code, coverage, bounds, spot-check escape probabilities |
| `vg demo` | all | the whole story: honest PASS, four tamper FAILs, cost summary |
| `vg bench` | all | overhead, retained bytes, opening bytes, verifier time, calibration quantiles |

## Project layout

```text
src/vgemma/
├── profile.py      Gemma 4 support checks (fail closed), per-layer shapes, config hash
├── model.py        checkpoint loading, text-decoder discovery by module name
├── canon.py        canonical bf16 arithmetic, soft-cap, exp, RoPE, norms, hashing
├── merkle.py       Merkle tree and proofs
├── protocol.py     leaves, IO chain, seeds, receipts, audits, opening container
├── keygen.py       weights / embedding roots, Freivalds vectors
├── tokenizer.py    checkpoint chat template, or the tiny byte-level tokenizer
├── tiny.py         tiny random Gemma 4 for CPU tests
├── auditor.py      challenges, client requests, HTTP client
├── demo.py · cli.py · display.py
├── prover/         hooks · engine (decode loop, commit, openings) · sampler · store · server · tamper
└── verifier/       verify · bindings · freivalds · bridge · attention · decode · key · codes · report
tests/              258 tests, all in tiny mode on CPU (plus a GPU smoke test)
infra/              modal_app.py · vast/
docs/               DECISIONS.md · SCHEMAS.md
```

## Documentation

| Document | Contents |
|---|---|
| [PRD.md](PRD.md) | goals, users, requirements, success criteria |
| [TECH_SPEC.md](TECH_SPEC.md) | protocol, capture plan, commitments, verifier order, tolerances |
| [docs/DECISIONS.md](docs/DECISIONS.md) | every departure from the draft spec, with the reason and the measurement behind it |
| [docs/SCHEMAS.md](docs/SCHEMAS.md) | receipt, challenge, opening, verdict, key and request formats |
| [AGENTS.md](AGENTS.md) | build order, non-negotiables and definition of done |

## Roadmap

- [x] **Tier 1 MVP:** the full protocol on `transformers` in bf16, the tiny CPU demo and suite,
  and the real 12B demo on an A100
- [ ] Gemma 4 31B on an 80 GB GPU
- [ ] **Tier 2, the exact INT8 path:** a W8A8 checkpoint, CommitLLM's vLLM sidecar and Rust
  verifier. Freivalds over a prime field makes the shell and bridges exact and removes the bf16
  tolerance.
- [ ] Compact retained state and faster commitment hashing
- [ ] Receipt transparency log and a receipt viewer

## Contributing

```bash
uv sync                       # CPU environment (macOS / Linux); `--extra gpu` for CUDA
uv run pytest -q              # must stay green; new checks come with an adversarial test
uv run ruff check && uv run ruff format --check
```

Ground rules, from [AGENTS.md](AGENTS.md):

- fail closed;
- no silent tolerance: every bound is printed with its measured deviation;
- the verifier never touches weights or a GPU;
- reason codes are listed in TECH_SPEC before they are used.

## Acknowledgements and licence

- The protocol design, vocabulary and adversarial scenario list come from
  [CommitLLM](https://github.com/lambdaclass/CommitLLM) by LambdaClass (MIT). No CommitLLM code is
  copied.
- Gemma 4 is by Google DeepMind (Apache 2.0). No weights are distributed here.
- See [NOTICE](NOTICE).

Released under the [MIT licence](LICENSE).
