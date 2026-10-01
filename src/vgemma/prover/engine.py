"""Prover engine: own decode loop with KV cache, capture, CPU soft-cap and sampling, commitments,
receipts and openings (TECH_SPEC sections 5, 6, 8).
"""

from __future__ import annotations

import json
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from vgemma import canon
from vgemma.canon import tensor_hash
from vgemma.merkle import MerkleTree
from vgemma.model import LoadedModel, checkpoint_files, iter_checkpoint_tensors, text_prefix, to_numpy
from vgemma.profile import GemmaProfile
from vgemma.protocol import (
    LM_HEAD_ID,
    OPENING_VERSION,
    RECEIPT_VERSION,
    SAMPLER_ID,
    SOFTCAP_ID,
    client_seed,
    derive_seed,
    embedding_leaves,
    encode_opening,
    io_chain,
    layer_leaf,
    logits_hash,
    normalize_audits,
    position_body,
    position_leaf,
    prompt_hash,
    required_tensors,
    seed_commitment,
    sign_receipt,
    tensor_key,
    weight_leaf,
    witness_hash,
)
from vgemma.prover.hooks import CaptureHooks
from vgemma.prover.sampler import SamplingPolicy, sample_step
from vgemma.prover.store import Commitment, RetainedStore, Trace
from vgemma.prover.tamper import Tamper
from vgemma.tokenizer import ChatTokenizer

# ---------------------------------------------------------------------------
# Prover identity (receipt signing key and seed secret; unrelated to the verifier key)
# ---------------------------------------------------------------------------


class ProverIdentity:
    def __init__(self, signing_seed: bytes, prover_secret: bytes):
        from nacl.signing import SigningKey

        self.signing_key = SigningKey(signing_seed)
        self.prover_secret = prover_secret

    @property
    def id(self) -> str:
        return "ed25519:" + self.signing_key.verify_key.encode().hex()

    @classmethod
    def load_or_create(cls, path: Path) -> ProverIdentity:
        path = Path(path)
        if path.exists():
            d = json.loads(path.read_text())
            return cls(bytes.fromhex(d["signing_seed"]), bytes.fromhex(d["prover_secret"]))
        seed, secret = secrets.token_bytes(32), secrets.token_bytes(32)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"signing_seed": seed.hex(), "prover_secret": secret.hex()}))
        path.chmod(0o600)
        return cls(seed, secret)


def compute_weights_root(model_dir: Path, rename: Callable[[str], str] | None = None, cache: Path | None = None) -> str:
    """Weights root over the checkpoint files (the same leaves keygen uses), optionally cached."""
    files = checkpoint_files(model_dir)
    stamp = json.dumps([[f.name, f.stat().st_size, f.stat().st_mtime_ns] for f in files] + [rename is not None])
    if cache and cache.exists():
        d = json.loads(cache.read_text())
        if d.get("stamp") == stamp and d.get("dir") == str(model_dir):
            return d["root"]
    if rename is None:
        leaves = [weight_leaf(n, a) for n, a in iter_checkpoint_tensors(model_dir)]
    else:
        content = dict(iter_checkpoint_tensors(model_dir))
        leaves = [weight_leaf(n, content[rename(n)]) for n in sorted(content)]
    root = MerkleTree(leaves).root.hex()
    if cache:
        cache.write_text(json.dumps({"stamp": stamp, "dir": str(model_dir), "root": root}))
    return root


# ---------------------------------------------------------------------------
# Commitment of a trace
# ---------------------------------------------------------------------------


def commit_trace(trace: Trace, profile: GemmaProfile) -> Commitment:
    """Layer leaves, position leaves, trace root and IO chain for a captured trace (no receipt yet)."""
    n_pos, n_groups = trace.n_positions, profile.num_layers + 1
    layer_leaves = np.zeros((n_pos, n_groups, 32), dtype=np.uint8)
    for layer in range(n_groups):
        arrs = [trace.layers[layer][n] for n in profile.group_names(layer)]
        for p in range(n_pos):
            leaf = layer_leaf(layer, p, [tensor_hash(a[p]) for a in arrs])
            layer_leaves[p, layer] = np.frombuffer(leaf, dtype=np.uint8)
    lhs = [logits_hash(trace.logits_precap[t]) for t in range(trace.n_gen)]
    whs = [witness_hash(w) for w in trace.witnesses]
    bodies = np.zeros((n_pos, 32), dtype=np.uint8)
    pos_leaves = np.zeros((n_pos, 32), dtype=np.uint8)
    for p in range(n_pos):
        t = trace.decode_step(p)
        body = position_body(
            [bytes(layer_leaves[p, layer]) for layer in range(n_groups)],
            lhs[t] if t is not None else None,
            trace.tokens[p + 1] if t is not None else None,
            whs[t] if t is not None else None,
        )
        bodies[p] = np.frombuffer(body, dtype=np.uint8)
        pos_leaves[p] = np.frombuffer(position_leaf(p, trace.tokens[p], body), dtype=np.uint8)
    return Commitment(
        receipt={},
        layer_leaves=layer_leaves,
        position_bodies=bodies,
        position_leaves=pos_leaves,
        logits_hashes=lhs,
        witness_hashes=whs,
    )


def build_receipt(
    trace: Trace, com: Commitment, model_info: dict[str, Any], manifest: dict[str, Any], identity: ProverIdentity
) -> dict[str, Any]:
    tree = MerkleTree([bytes(x) for x in com.position_leaves])
    p_hash = prompt_hash(trace.tokens[: trace.n_prompt])
    transcript = [(trace.tokens[trace.n_prompt + t], com.logits_hashes[t]) for t in range(trace.n_gen)]
    receipt = {
        "version": RECEIPT_VERSION,
        "request_id": trace.request_id,
        "model": dict(model_info),
        "manifest": dict(manifest),
        "prompt_hash": p_hash.hex(),
        "seed_commitment": seed_commitment(trace.seed, trace.request_id).hex(),
        "client_nonce": trace.client_nonce.hex() if trace.client_nonce else None,
        "n_prompt": trace.n_prompt,
        "n_gen": trace.n_gen,
        "trace_root": tree.root.hex(),
        "io_chain_head": io_chain(p_hash, transcript).hex(),
        "prover": {"id": identity.id, "signature": ""},
    }
    return sign_receipt(receipt, identity.signing_key)


# ---------------------------------------------------------------------------
# Openings
# ---------------------------------------------------------------------------


def build_opening(
    trace: Trace,
    com: Commitment,
    challenge: dict[str, Any],
    profile: GemmaProfile,
    embed_bits: np.ndarray,
    embed_tree: MerkleTree,
) -> bytes:
    """Open what the challenge's audits need, and bind the input token of every other position."""
    audits = normalize_audits(challenge)
    n_pos, n_layers = trace.n_positions, profile.num_layers
    if any(not 0 <= a["pos"] < n_pos for a in audits):
        raise ValueError(f"audit positions must lie in [0, {n_pos})")
    if any(not 0 <= x < n_layers for a in audits for x in a["layers"]):
        raise ValueError(f"audit layers must lie in [0, {n_layers})")
    if not any(a["layers"] for a in audits):
        raise ValueError("a challenge must audit at least one layer")
    audited = {a["pos"] for a in audits}
    req = required_tensors(profile, trace.n_prompt, trace.n_gen, audits)
    tree = MerkleTree([bytes(x) for x in com.position_leaves])
    tensors: dict[str, np.ndarray] = {}
    entries: dict[str, Any] = {}
    for j in range(n_pos):
        proof = [h.hex() for h in tree.proof(j)]
        if j not in req:  # token-only: binds the input token to the trace root
            body = bytes(com.position_bodies[j]).hex()
            entries[str(j)] = {"input_token": trace.tokens[j], "body": body, "proof": proof}
            continue
        t = trace.decode_step(j)
        entry: dict[str, Any] = {
            "input_token": trace.tokens[j],
            "out_token": trace.tokens[j + 1] if t is not None else None,
            "proof": proof,
            "layers": {},
        }
        if t is None:
            entry["logits"], entry["witness"] = None, None
        elif j in audited:
            entry["logits"], entry["witness"] = "open", trace.witnesses[t]
            tensors[f"p{j}/logits_precap"] = trace.logits_precap[t]  # post-cap is recomputed by the verifier
        else:
            entry["logits"], entry["witness"] = com.logits_hashes[t].hex(), com.witness_hashes[t].hex()
        for layer in range(n_layers + 1):
            need = req[j].get(layer)
            if not need:
                entry["layers"][str(layer)] = {"leaf": bytes(com.layer_leaves[j, layer]).hex()}
                continue
            group = {}
            for name in profile.group_names(layer):
                row = trace.layers[layer][name][j]
                if name in need:
                    group[name] = "open"
                    tensors[tensor_key(j, layer, name)] = row
                else:
                    group[name] = tensor_hash(row).hex()
            entry["layers"][str(layer)] = {"tensors": group}
        if j in audited:
            tok = trace.tokens[j]
            entry["embedding"] = {"proof": [h.hex() for h in embed_tree.proof(tok)]}
            tensors[f"p{j}/embed_row"] = embed_bits[tok]
        entries[str(j)] = entry
    index = {
        "version": OPENING_VERSION,
        "request_id": trace.request_id,
        "challenge": {"audits": audits, "tier": challenge.get("tier", "custom")},
        "seed": trace.seed.hex(),
        "prompt_tokens": trace.tokens[: trace.n_prompt],
        "io_transcript": [[trace.tokens[trace.n_prompt + t], com.logits_hashes[t].hex()] for t in range(trace.n_gen)],
        "positions": entries,
    }
    return encode_opening(index, tensors)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


@dataclass
class GenerationResult:
    receipt: dict[str, Any]
    text: str
    token_ids: list[int]
    trace: Trace
    timings: dict[str, float] = field(default_factory=dict)


class Engine:
    def __init__(
        self,
        lm: LoadedModel,
        tokenizer: ChatTokenizer,
        store: RetainedStore,
        identity: ProverIdentity,
        public: dict[str, Any] | None = None,
        tamper: Tamper | None = None,
        log=print,
    ):
        self.lm, self.tokenizer, self.store, self.identity, self.tamper, self.log = (
            lm,
            tokenizer,
            store,
            identity,
            tamper,
            log,
        )
        self.profile = lm.profile
        if public is not None and public["config_hash"] != self.profile.config_hash():
            raise ValueError("served checkpoint config hash differs from the public params (fail closed)")
        files_root = compute_weights_root(lm.model_dir, cache=store.root / "weights_root_cache.json")
        self.tamper_info: dict[str, Any] = {}
        if tamper is not None:
            tamper.warn(log)
            self.tamper_info = tamper.apply_to_model(lm, log)
        if tamper is not None and tamper.mode == "identity-root":
            names = [n for n, _ in iter_checkpoint_tensors(lm.model_dir)]
            prefix = text_prefix(names)
            self.weights_root = compute_weights_root(
                lm.model_dir, rename=lambda n: tamper.renamed_tensor(n, prefix, lm)
            )
        else:
            self.weights_root = files_root  # weights / identity tampers keep reporting the pristine root
        self.embed_bits = to_numpy(lm.text_model.embed_tokens.weight)
        self.embed_tree = MerkleTree(embedding_leaves(self.embed_bits))
        if public is not None and tamper is None:
            if public["weights_root"] != self.weights_root:
                log("WARNING: served checkpoint weights root differs from public params")
            if public["embedding_root"] != self.embed_tree.root.hex():
                log("WARNING: served embedding root differs from public params")
        import torch

        torch.backends.cuda.matmul.allow_tf32 = False  # the LM-head bound assumes true f32 accumulation
        torch.set_float32_matmul_precision("highest")
        self.lm_head_f32 = lm.lm_head.weight.detach().float()  # [vocab, hidden], the head actually served
        self.capture = CaptureHooks(lm)
        self.capture.install()

    # -- receipts -------------------------------------------------------------------------------

    @property
    def model_info(self) -> dict[str, Any]:
        return {
            "id": self.lm.model_id,
            "revision": self.lm.revision,
            "weights_root": self.weights_root,
            "config_hash": self.profile.config_hash(),
        }

    def manifest(self, policy: SamplingPolicy, thinking: bool, max_new_tokens: int) -> dict[str, Any]:
        p = self.profile
        return {
            "dtype": "bfloat16",
            "attn_implementation": self.lm.attn_implementation,
            "temperature": float(policy.temperature),
            "top_p": float(policy.top_p),
            "top_k": int(policy.top_k),
            "greedy": bool(policy.greedy),
            "final_logit_softcapping": p.final_logit_softcapping,
            "embed_scale_bf16": p.embed_scale_bf16,
            "rms_norm_eps": p.rms_norm_eps,
            "sliding_window": p.sliding_window,
            "layer_types_hash": p.layer_types_hash(),
            "attention_k_eq_v": p.attention_k_eq_v,
            "rope_hash": p.rope_hash(),
            "qk_norm": True,
            "thinking": bool(thinking),
            "chat_template_hash": self.tokenizer.chat_template_hash,
            "tokenizer_hash": self.tokenizer.tokenizer_hash,
            "speculative": "none",
            "prefix_caching": False,
            "max_new_tokens": int(max_new_tokens),
            "eos_token_ids": sorted(self.lm.eos_token_ids),
            "sampler": SAMPLER_ID,
            "softcap_impl": SOFTCAP_ID,
            "lm_head": LM_HEAD_ID,
        }

    # -- generation -----------------------------------------------------------------------------

    def generate(
        self,
        messages: list[dict[str, str]] | None = None,
        prompt_tokens: list[int] | None = None,
        max_new_tokens: int = 32,
        policy: SamplingPolicy | None = None,
        thinking: bool = False,
        request_id: str | None = None,
        capture: bool = True,
        retain: bool = True,
        client_nonce: bytes | None = None,
    ) -> GenerationResult:
        """With ``client_nonce`` the sampling randomness is ``H(nonce)``: the client fixes it and the
        prover cannot try seeds until it likes the output. Without one it comes from the prover secret."""
        policy = policy or SamplingPolicy()
        if prompt_tokens is None:
            if messages is None:
                raise ValueError("messages or prompt_tokens required")
            prompt_tokens = self.tokenizer.apply_chat(messages, thinking=thinking)
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        request_id = request_id or "r_" + secrets.token_hex(8)
        if client_nonce is not None and len(client_nonce) != 32:
            raise ValueError("client nonce must be 32 bytes")
        seed = client_seed(client_nonce) if client_nonce else derive_seed(self.identity.prover_secret, request_id)
        actual = self.tamper.sampling_policy(policy) if self.tamper else policy
        t0 = time.perf_counter()
        gen, pre, post, wits, layers, timing = self._decode(prompt_tokens, max_new_tokens, actual, seed, capture)
        trace = Trace(
            request_id=request_id,
            tokens=list(prompt_tokens) + gen,
            n_prompt=len(prompt_tokens),
            n_gen=len(gen),
            layers=layers,
            logits_precap=np.stack(pre),
            logits_postcap=np.stack(post),
            witnesses=wits,
            seed=seed,
            client_nonce=client_nonce,
        )
        timing["generate_s"] = time.perf_counter() - t0
        receipt: dict[str, Any] = {}
        if capture:
            t1 = time.perf_counter()
            com = commit_trace(trace, self.profile)
            receipt = build_receipt(
                trace, com, self.model_info, self.manifest(policy, thinking, max_new_tokens), self.identity
            )
            com.receipt = receipt
            timing["commit_s"] = time.perf_counter() - t1
            if retain:
                t2 = time.perf_counter()
                timing["retained_bytes"] = self.store.save(trace, com)
                timing["store_s"] = time.perf_counter() - t2
        return GenerationResult(receipt, self.tokenizer.decode(gen), gen, trace, timing)

    def _logits(self, h_final) -> np.ndarray:
        """Pre-cap logits in f32 from the final hidden state (the protocol's LM head, ``LM_HEAD_ID``)."""
        return (self.lm_head_f32 @ h_final.float()).cpu().numpy()

    def _decode(self, prompt: list[int], max_new: int, policy: SamplingPolicy, seed: bytes, capture: bool):
        import torch
        from transformers import DynamicCache

        lm, p = self.lm, self.profile
        dev = lm.device
        eos = set(lm.eos_token_ids)
        self.capture.reset()
        self.capture.enabled = capture
        gen: list[int] = []
        pre, post, wits = [], [], []
        timing: dict[str, float] = {}
        try:
            with torch.inference_mode():
                cache = DynamicCache(config=lm.text_model.config)
                n = len(prompt)
                t0 = time.perf_counter()
                out = lm.text_model(
                    input_ids=torch.tensor([prompt], device=dev),
                    position_ids=torch.arange(n, device=dev)[None],
                    past_key_values=cache,
                    use_cache=True,
                )
                h = out.last_hidden_state[:, -1:]
                t_prefill = time.perf_counter()
                for t in range(max_new):
                    precap = self._logits(h[0, -1])
                    honest = canon.softcap(precap, p.final_logit_softcapping)
                    postcap = self.tamper.postcap(precap, honest) if self.tamper else honest
                    token, wit = sample_step(postcap, policy, seed, t)
                    gen.append(token)
                    pre.append(precap)
                    post.append(postcap)
                    wits.append(wit)
                    if token in eos or t == max_new - 1:
                        break
                    out = lm.text_model(
                        input_ids=torch.tensor([[token]], device=dev),
                        position_ids=torch.tensor([[n + t]], device=dev),
                        past_key_values=cache,
                        use_cache=True,
                    )
                    h = out.last_hidden_state[:, -1:]
                if dev.startswith("cuda"):
                    torch.cuda.synchronize()
                t_end = time.perf_counter()
            timing.update(prefill_s=t_prefill - t0, decode_s=t_end - t_prefill, n_gen=len(gen), n_prompt=n)
            layers = self.capture.collect(n + len(gen) - 1) if capture else {}
        finally:
            self.capture.enabled = False
            self.capture.reset()
        return gen, pre, post, wits, layers, timing

    # -- audit ------------------------------------------------------------------------------------

    def open(self, request_id: str, challenge: dict[str, Any]) -> bytes:
        trace, com = self.store.load(request_id)
        return build_opening(trace, com, challenge, self.profile, self.embed_bits, self.embed_tree)

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "model": self.model_info,
            "embedding_root": self.embed_tree.root.hex(),
            "prover_id": self.identity.id,
            "attn_implementation": self.lm.attn_implementation,
            "retained_requests": self.store.count(),
            "tamper": self.tamper.mode if self.tamper else None,
            "profile": self.profile.summary(),
        }


def measure_overhead(
    engine: Engine, messages: list[dict[str, str]], max_new_tokens: int, runs: int = 3
) -> dict[str, Any]:
    """Tokens per second with capture off and on for the same request (greedy, nothing retained).

    ``decode_tok_s`` counts the forward passes only; ``serve_tok_s`` also includes moving captures
    to the CPU and committing (hashing, Merkle tree, receipt signature).
    """
    policy = SamplingPolicy(greedy=True)
    out: dict[str, Any] = {}
    for capture in (False, True):
        rows = []
        for _ in range(runs):
            t0 = time.perf_counter()
            res = engine.generate(
                messages=messages, max_new_tokens=max_new_tokens, policy=policy, capture=capture, retain=False
            )
            total = time.perf_counter() - t0
            t = res.timings
            rows.append((t["n_gen"] / (t["prefill_s"] + t["decode_s"]), t["n_gen"] / total, t.get("commit_s", 0.0)))
        rows.sort()
        mid = rows[len(rows) // 2]
        out["capture" if capture else "plain"] = {
            "decode_tok_s": mid[0],
            "serve_tok_s": mid[1],
            "commit_s": mid[2],
            "n_gen": res.timings["n_gen"],
            "n_prompt": res.timings["n_prompt"],
        }
    plain, cap = out["plain"], out["capture"]
    out["decode_overhead"] = 1.0 - cap["decode_tok_s"] / plain["decode_tok_s"]
    out["serve_overhead"] = 1.0 - cap["serve_tok_s"] / plain["serve_tok_s"]
    return out
