# AGENTS.md: building verifiable-gemma

You are building `verifiable-gemma`: a commit-and-audit verification layer for Gemma 4 inference, following the CommitLLM protocol. Read `PRD.md` for what and why, `TECH_SPEC.md` for how. This file tells you what to do first, what to skip, and what "done" means.

## Mission, in order

1. An end-to-end MVP that runs on CPU in tiny mode: `vg demo --tiny` prints an honest PASS and four tamper FAILs with the correct reason codes, with every check listed in TECH_SPEC section 10 implemented.
2. The same demo on a real Gemma 4 checkpoint on one GPU (`google/gemma-4-12B-it` by default, `31B-it` on an 80 GB GPU).
3. The adversarial test suite green in tiny mode.
4. Metrics printed by `vg demo` and `vg bench`, written into README.
5. Everything else (signatures, receipt viewer, Tier 2 exact INT8 path) only after 1 to 4 are done.

Do not start Tier 2 (vLLM sidecar, Rust verifier, W8A8 checkpoint) until `vg demo` passes on a real model. Tier 2 is in TECH_SPEC section 15 so the MVP's data model lines up with it; it is not the first deliverable.

## Build order

Work in vertical slices. Each step ends with a runnable command and a test.

1. `profile.py` + `canon.py` + `merkle.py` + tests. Canonical functions (`rmsnorm_gemma`, `gelu_tanh`, `softcap`, embed scale rounding, RoPE tables for sliding and global layers) must match `transformers` Gemma 4 modules on random inputs before anything else is built on them.
2. `vg tiny`: build the tiny Gemma 4 checkpoint directory (TECH_SPEC section 12). Confirm `transformers` loads it and generates.
3. `vg keygen` on the tiny checkpoint: weights root, embedding root, Freivalds vectors with per-layer-type families, norm weights, config hash. Assert global layers have no Wv family.
4. `prover/hooks.py` + `prover/engine.py`: own decode loop with KV cache, capture per position, CPU sampler, commitments, receipt. Run one request in tiny mode and dump the receipt.
5. `verifier/`: Merkle and bindings first, then decode (soft-cap, sampling, LM-head binding), then Freivalds per family, then bridge, then wiring and provenance, then attention replay. After each component, add the matching tamper or bit-flip test and watch it fail for the right reason.
6. `prover/server.py` + `vg chat` + `vg audit` + `vg verify` over HTTP.
7. Tamper modes in the engine, `vg demo --tiny`.
8. Real model: run `vg keygen` and `vg demo` on the GPU model. Fix shape or naming surprises in `model.py` (module discovery by regex), not by special-casing the verifier.
9. `vg bench`, README metrics table, NOTICE.

## Non-negotiables

- Fail closed. Unsupported configs (MoE enabled, `hidden_size_per_layer_input > 0`, `num_kv_shared_layers > 0`, `use_double_wide_mlp`, unknown `layer_types`, `attention_k_eq_v` false when the profile expects true) raise at keygen and at serve with a clear message.
- No silent tolerance. Every tolerance-bounded check prints the observed deviation and the bound. Exact checks are compared exactly. Audit-only checks are labelled "audited" in the coverage table.
- The verifier never loads model weights and never touches a GPU. If a check seems to need the weights, it needs a Freivalds vector or a Merkle proof instead.
- The prover and verifier share one sampler implementation and one set of canonical functions (`vgemma/canon.py`, `vgemma/prover/sampler.py` imported by both sides). Never duplicate them.
- Reason codes are the ones listed in TECH_SPEC section 10. Add new ones to the list in the spec before using them.
- The provider must never see the key. Keep `key.npz` out of the server process; the server only reads `public.json`.
- Tamper modes exist only behind an explicit `--tamper` flag and log a loud warning.
- Honesty in docs: the README carries the verified / audited / open table; do not describe tolerance-bounded checks as exact, and do not describe attention replay as verification.

## Project tooling: uv, and nothing else

This is a `uv` project. Do not use pip, poetry, conda, requirements.txt or setup.py anywhere, including inside Modal images.

- `uv init --package verifiable-gemma` layout: `pyproject.toml` with `[project]`, `[project.scripts] vg = "vgemma.cli:app"`, `[build-system]` using `hatchling`, `[dependency-groups] dev = [pytest, ruff, pytest-timeout]`, `[project.optional-dependencies] gpu = [...]`, `modal = ["modal"]`.
- `.python-version` pinned to `3.12`. `uv.lock` is committed and kept in sync (`uv lock` after every dependency change; CI runs `uv sync --locked`).
- Runtime dependencies: `torch`, `transformers` (a release that includes `gemma4`), `safetensors`, `numpy`, `fastapi`, `uvicorn`, `httpx`, `typer`, `pynacl`, `huggingface-hub`, `accelerate`.
- Torch index: configure `[tool.uv.sources]` and `[[tool.uv.index]]` so that `uv sync` on a CPU machine gets the CPU wheel and `uv sync --extra gpu` gets the CUDA wheel (use the `pytorch-cu12x` index with `marker`-based selection as documented by uv). The tiny-mode test suite must install and run on a laptop without CUDA.
- Every command goes through `uv run`: `uv run vg ...`, `uv run pytest`, `uv run ruff check`. Never activate a venv by hand in scripts.
- Keep modules small and typed. No frameworks beyond the list. No async where sync is enough.
- Tensor capture stays in the model dtype on device until commit; move to CPU in bulk at commit time, not per hook call, except logits (f32, per step).
- Hash with `hashlib.sha256`; domain-separate every hash (TECH_SPEC section 6).
- Canonical JSON: `json.dumps(obj, sort_keys=True, separators=(",", ":"))` for anything hashed.
- Tests under `tests/` run in tiny mode on CPU and finish fast. Mark real-model tests with `@pytest.mark.gpu` and skip them when no CUDA device is present.
- `ruff check` and `ruff format` clean before every commit. Commit small, message in the imperative.

## Working with CommitLLM code

- The protocol and its vocabulary come from https://github.com/lambdaclass/CommitLLM (MIT). Add `NOTICE` on day one with the repository URL, licence and the upstream commit you read.
- Copying is allowed. When you copy or adapt a file or function, keep the MIT header, and add a line to `NOTICE`: upstream path, local path, what changed.
- Prefer reading CommitLLM for structure (receipt fields, challenge tiers, adversarial scenario list, reason-code style) and writing the Python fresh; the Rust crates and the vLLM sidecar are for Tier 2.
- Useful upstream references: `README.md` (coverage table and claim wording), `roadmap.md` (items 25, 31, 34), `docs/security/redteam_audit_only.md` (residual hole), `scripts/modal/tests/llama/test_adversarial.py` (the 36 scenarios to mirror), `sidecar/verilm/hooks.py` and `server.py` (capture and manifest fields).

## GPU infrastructure: Modal first, vast.ai as backup

All GPU work runs on Modal from code in `infra/modal_app.py`. Nothing GPU-related depends on a shell session on a remote box.

- One Modal app `verifiable-gemma`. Image: `modal.Image.debian_slim(python_version="3.12").uv_sync(extras=["gpu"])` built from this repo's `pyproject.toml` and `uv.lock` (fall back to `.uv_pip_install(...)` from the locked list only if `uv_sync` is unavailable in the installed `modal` version). Add the repo as `add_local_python_source("vgemma")` so the code is the one being edited.
- Volumes: `vg-hf-cache` mounted at `/cache` with `HF_HOME=/cache/hf` (model weights download once); `vg-state` at `/state` for receipts, openings, retained state and `public.json`; `vg-keys` at `/keys` for secret keys. `vg-keys` is mounted only in `keygen` and `verify` functions, never in `serve`. This is the key-separation rule enforced by infrastructure, not by convention.
- Functions: `keygen` (CPU, `memory=65536`, mounts `/cache`, `/state`, `/keys`); `demo` (GPU; runs the server as a subprocess with `/state` only, then runs the auditor and verifier in the main process with `/keys`); `bench` (GPU); `serve` (GPU, `@modal.web_server` on port 8000, mounts `/cache` and `/state` only); `tests_tiny` (CPU, runs `pytest`); `download` (CPU, pre-pulls a model into the cache).
- GPU choice: `gpu="A100-40GB"` or `"L40S"` for `gemma-4-12B-it` (24 GB bf16); `gpu="A100-80GB"` or `"H100"` for `gemma-4-31B-it` (62 GB bf16). Set `timeout` generously (keygen on 31B reads 62 GB) and `scaledown_window` short.
- Commands: `uv run modal run infra/modal_app.py::demo --model google/gemma-4-12B-it`, `uv run modal run --detach infra/modal_app.py::keygen --model ...`, `uv run modal deploy infra/modal_app.py` for the web server, `uv run modal volume ls vg-state` to inspect outputs, `uv run modal app logs verifiable-gemma`. Use `--detach` for anything over a few minutes and poll logs.
- Local loop: every protocol change is validated with `uv run vg demo --tiny` and `uv run pytest` on CPU before any Modal run. Modal is for real-model smoke, demo and bench only.
- Secrets: none required (Gemma 4 is ungated). If a `HF_TOKEN` is ever needed, use a Modal secret, never an environment file in the repo.

Backup: vast.ai. `infra/vast/setup.sh` provisions a rented instance (CUDA 12 PyTorch image, 1 GPU of the same class as above, 200 GB disk): installs `uv`, clones the repo, runs `uv sync --extra gpu`, sets `HF_HOME` on the instance disk, and runs the same `uv run vg ...` commands over SSH. `infra/vast/README.md` lists the instance filters (GPU, VRAM, disk, bandwidth) and the one-line run commands. The code path is identical; only the launcher differs.

## Model and hardware notes

- `google/gemma-4-12B-it` and `google/gemma-4-31B-it` are ungated and Apache 2.0; no token required. Downloads are 24 GB and 62 GB in bf16. Prefer 12B unless an 80 GB GPU is available. Pre-pull with `infra/modal_app.py::download` so the demo function does not spend its GPU time downloading.
- Load the published checkpoint as is; pass text only. Find the text decoder and its layers by name pattern (`layers.{i}.input_layernorm`, `layers.{i}.self_attn.q_proj`, ...). Assert the discovered layer count equals `text_config.num_hidden_layers` and that global layers have no `v_proj`.
- Use `attn_implementation="sdpa"` (or `"eager"` if SDPA misbehaves with hooks); record it in the manifest. Do not enable any speculative or assisted generation.
- Sampling defaults from the model card: temperature 1.0, top_p 0.95, top_k 64. Thinking off by default.
- bf16 on GPU. On CPU tiny mode, run bf16 too if the installed torch supports it, so rounding behaviour matches; otherwise float32 with the tolerance recomputed for f32 and printed.

## Definition of done per component

| Component | Done when |
|---|---|
| canon | Unit tests against `transformers` modules pass for every function, both layer types |
| merkle | Proof round-trip tests pass; tampered sibling rejected |
| keygen | Runs on tiny and on the real model; key size printed; families per layer type correct |
| engine | Receipt produced; retained state written; re-running the same request with the same seed reproduces the same tokens |
| verifier | Honest tiny run PASS with full and routine layers; every adversarial scenario FAIL with expected code |
| server | `vg chat`, `vg audit`, `vg verify` work over HTTP against tiny |
| demo | `vg demo --tiny` and `vg demo` on GPU print PASS then four FAILs with correct codes and the cost summary |
| bench | Numbers printed and copied into README |

## When something is unclear

- Prefer the simpler mechanism that keeps the check honest. For example, if post-RoPE Q and K cannot be captured, capture post-norm tensors and recompute RoPE on the verifier; do not drop the attention audit.
- If a Gemma 4 module name differs between model sizes, extend the regex in `model.py`; never hardcode a size.
- If a tolerance has to be relaxed to make an honest run pass, print the measured deviation distribution, set the bound from the measurement with a stated margin, and note it in the README. Never relax silently.
- If the real model does not fit, fall back to 12B or to shorter contexts; do not disable capture for some layers.
- Keep a `docs/DECISIONS.md` log: one line per decision with the reason.

## Repository layout

```
verifiable-gemma/
  AGENTS.md  PRD.md  TECH_SPEC.md  README.md  NOTICE  LICENSE
  pyproject.toml  uv.lock  .python-version  ruff.toml  .gitignore
  src/vgemma/
    __init__.py  profile.py  model.py  canon.py  merkle.py  keygen.py  cli.py
    prover/   hooks.py  sampler.py  engine.py  store.py  server.py  tamper.py
    verifier/ freivalds.py  bridge.py  attention.py  decode.py  bindings.py  verify.py  codes.py
  tests/      test_canon.py  test_merkle.py  test_keygen.py  test_protocol_tiny.py  test_adversarial_tiny.py  test_gpu_smoke.py
  infra/      modal_app.py  vast/setup.sh  vast/README.md
  docs/       DECISIONS.md  SCHEMAS.md
```

`src/` layout so that `uv run` always imports the installed package, not a stray working-directory copy. `.gitignore` excludes `keys/`, `state/`, `*.npz`, `*.safetensors`, `.venv/`, `__pycache__/`, `.modal/`.

## First commands

```
uv init --package verifiable-gemma --python 3.12   # if starting from empty
uv add torch transformers safetensors numpy fastapi uvicorn httpx typer pynacl huggingface-hub accelerate
uv add --optional modal modal
uv add --dev pytest pytest-timeout ruff
uv sync
uv run vg tiny --out ./tiny-gemma4
uv run vg keygen --model ./tiny-gemma4 --out ./keys/tiny
uv run vg demo --tiny
uv run pytest -q
uv run modal run infra/modal_app.py::download --model google/gemma-4-12B-it
uv run modal run --detach infra/modal_app.py::keygen --model google/gemma-4-12B-it
uv run modal run infra/modal_app.py::demo --model google/gemma-4-12B-it
```
