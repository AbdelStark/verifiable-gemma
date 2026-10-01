"""Verifier orchestration (TECH_SPEC section 10): receipt + opening + key + challenge -> verdict.

No GPU, no model weights: every check uses the opening, the Freivalds vectors and norm weights in
the key, Merkle roots, and the shared canonical functions and sampler. The challenge is the
auditor's own (never the prover's echo), and every reference value comes from the key's metadata.
Fail fast: the first failing check gives the reason code; the coverage table and the
deviation/bound of every check that ran are always reported.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import numpy as np

from vgemma.canon import (
    F32_U,
    bf16_mul,
    bf16_to_f64,
    bf16_ulp_distance,
    f32_to_bf16,
    rmsnorm_gemma,
    softcap,
    tensor_hash,
)
from vgemma.merkle import verify_proof
from vgemma.profile import GemmaProfile
from vgemma.protocol import (
    OPENING_VERSION,
    decode_opening,
    decode_step,
    layer_leaf,
    logits_hash,
    normalize_audits,
    position_body,
    position_leaf,
    required_tensors,
    tensor_key,
    witness_hash,
)
from vgemma.verifier import attention, bridge, freivalds
from vgemma.verifier.bindings import (
    check_expected,
    check_io_chain,
    check_manifest,
    check_prompt,
    check_receipt_schema,
    check_roots,
    check_seed,
    check_signature,
)
from vgemma.verifier.codes import SHELL
from vgemma.verifier.decode import policy_from_manifest, replay_token
from vgemma.verifier.key import VerifierKey
from vgemma.verifier.report import Ledger, VerifyFail

VERDICT_VERSION = 1
BF16_EXP = np.uint16(0x7F80)

# (family, output tensor, input tensor)
SHELL_IO = (
    ("wq", "q", "x_attn"),
    ("wk", "k", "x_attn"),
    ("wv", "v", "x_attn"),
    ("wo", "o", "a"),
    ("wgate", "g", "x_ffn"),
    ("wup", "u", "x_ffn"),
    ("wdown", "d", "h"),
)

COMPONENTS: dict[str, tuple[str, ...]] = {
    "bindings": (
        "RECEIPT_SCHEMA", "RECEIPT_SIGNATURE", "OPENING_SCHEMA", "WEIGHTS_ROOT", "CONFIG_HASH",
        "MANIFEST_UNSUPPORTED", "MANIFEST_MISMATCH", "SEED_COMMITMENT", "PROMPT_BINDING", "IO_CHAIN",
        "MERKLE_POSITION",
    ),
    "embedding": ("EMBEDDING",),
    "shell": tuple(SHELL.values()),
    "bridge": (
        "BRIDGE_NORM_INPUT", "BRIDGE_NORM_Q", "BRIDGE_NORM_K", "BRIDGE_NORM_V", "KV_SHARED",
        "BRIDGE_NORM_POST_ATTN", "BRIDGE_RESIDUAL", "BRIDGE_NORM_PRE_FFN", "BRIDGE_GELU",
        "BRIDGE_NORM_POST_FFN", "BRIDGE_NORM_FINAL",
    ),
    "attention": ("WIRING", "KV_PROVENANCE", "ATTN_REPLAY"),
    "decode": ("LMHEAD_BINDING", "DECODE_SOFTCAP", "DECODE_SAMPLING"),
}  # fmt: skip


@contextmanager
def stage(code: str, position: int | None = None, layer: int | None = None):
    """Any unexpected error while checking malformed data fails closed with this stage's code."""
    try:
        yield
    except VerifyFail:
        raise
    except Exception as e:  # noqa: BLE001
        raise VerifyFail(code, f"malformed data: {type(e).__name__}: {e}", layer, position) from e


def _finite(a: np.ndarray) -> bool:
    if a.dtype == np.uint16:
        return not bool(np.any((a & BF16_EXP) == BF16_EXP))  # bf16 inf or NaN
    return bool(np.all(np.isfinite(a)))


class _Run:
    def __init__(self, receipt, opening, key: VerifierKey, public, challenge, prover_id, expected):
        self.receipt, self.opening, self.key, self.public = receipt, opening, key, public
        self.challenge, self.prover_id, self.expected = challenge, prover_id, expected or {}
        self.profile: GemmaProfile = key.profile
        self.ledger = Ledger()
        self.audits: list[dict[str, Any]] = []
        self.tensors: dict[str, np.ndarray] = {}

    # -- helpers ------------------------------------------------------------------------------

    def t(self, pos: int, layer: int, name: str) -> np.ndarray:
        return self.tensors[tensor_key(pos, layer, name)]

    def f(self, pos: int, layer: int, name: str) -> np.ndarray:
        return bf16_to_f64(self.t(pos, layer, name))

    def tol(self, code: str, fn: Callable[[], tuple[float, float] | float], layer=None, pos=None, unit="") -> None:
        """Run one tolerance check under its own stage code; ``fn`` returns (deviation, bound) or a ratio."""
        with stage(code, pos, layer):
            out = fn()
            dev, bound = out if isinstance(out, tuple) else (out, 1.0)
            self.ledger.tolerance(code, dev, bound, layer, pos, unit=unit)

    @property
    def positions(self) -> list[int]:
        return [a["pos"] for a in self.audits]

    @property
    def opened_decode(self) -> list[int]:
        """Audited positions that sample a token: their logits are opened and the token is checked."""
        r = self.receipt
        return [p for p in self.positions if decode_step(r["n_prompt"], r["n_gen"], p) is not None]

    # -- stages ------------------------------------------------------------------------------

    def run(self) -> None:
        L = self.ledger
        with stage("RECEIPT_SCHEMA"):
            check_receipt_schema(self.receipt, L)
        r = self.receipt
        with stage("RECEIPT_SIGNATURE"):
            check_signature(r, L, self.prover_id)
        with stage("OPENING_SCHEMA"):
            self._parse_opening()
        with stage("WEIGHTS_ROOT"):
            check_roots(r, self.public, self.key.meta, L)
        with stage("MANIFEST_UNSUPPORTED"):
            check_manifest(r["manifest"], self.profile, self.key.meta, L)  # reference values from the key
        with stage("MANIFEST_MISMATCH"):
            check_expected(r, self.expected, L)
        with stage("SEED_COMMITMENT"):
            self.seed = check_seed(r, self.index.get("seed"), L)
        with stage("PROMPT_BINDING"):
            self.prompt = check_prompt(r, self.index.get("prompt_tokens"), self.expected, L)
        with stage("IO_CHAIN"):
            self.transcript = check_io_chain(r, self.index.get("io_transcript"), L)
        self._check_structure()
        # audited positions first, then K/V provenance rows, then token-only positions
        audited = set(self.positions)
        for j in sorted(self.entries, key=lambda j: (j not in audited, "layers" not in self.entries[j], j)):
            self._check_position_leaf(j, j in audited)
        for a in self.audits:
            self._check_audit(a)
        for p in self.opened_decode:
            self._check_decode(p, decode_step(r["n_prompt"], r["n_gen"], p))

    def _parse_opening(self) -> None:
        L, r, prof = self.ledger, self.receipt, self.profile
        if isinstance(self.opening, (bytes, bytearray)):
            self.index, self.tensors = decode_opening(bytes(self.opening))
        else:
            self.index, self.tensors = self.opening
        idx = self.index
        L.exact("OPENING_SCHEMA", idx.get("version") == OPENING_VERSION, f"opening version {idx.get('version')!r}")
        L.exact("OPENING_SCHEMA", idx.get("request_id") == r["request_id"], "opening is for another request")
        L.exact("OPENING_SCHEMA", isinstance(self.challenge, dict), "the auditor's challenge is required")
        self.audits = normalize_audits(self.challenge)
        L.exact(
            "OPENING_SCHEMA",
            normalize_audits(idx["challenge"]) == self.audits,
            "opening answers a different challenge than the auditor's",
        )
        self.n_pos = r["n_prompt"] + r["n_gen"] - 1
        L.exact("OPENING_SCHEMA", any(a["layers"] for a in self.audits), "no audited layers")
        L.exact("OPENING_SCHEMA", all(0 <= p < self.n_pos for p in self.positions), "audit position out of range")
        L.exact(
            "OPENING_SCHEMA",
            all(0 <= x < prof.num_layers for a in self.audits for x in a["layers"]),
            "audit layer out of range",
        )
        self.entries = {int(k): v for k, v in idx["positions"].items()}
        L.exact("OPENING_SCHEMA", len(self.entries) == len(idx["positions"]), "duplicate position keys")
        bad = [k for k, a in self.tensors.items() if not _finite(a)]
        L.exact("OPENING_SCHEMA", not bad, f"non-finite values in opened tensors {bad[:4]}")

    def _check_structure(self) -> None:
        """Entry kinds, wiring of every opened layer group (names and shapes per layer type), then
        completeness: every tensor the challenge requires is opened."""
        L, prof, r = self.ledger, self.profile, self.receipt
        L.exact(
            "OPENING_SCHEMA",
            sorted(self.entries) == list(range(self.n_pos)),
            "every position must be opened, at least to bind its input token",
        )
        req = required_tensors(prof, r["n_prompt"], r["n_gen"], self.audits)
        for j, e in sorted(self.entries.items()):
            if "layers" not in e:
                L.exact(
                    "OPENING_SCHEMA",
                    set(e) == {"input_token", "body", "proof"} and j not in req,
                    f"position {j} is token-only but the challenge needs its tensors",
                    position=j,
                )
                continue
            with stage("WIRING", j):
                n_groups = len(e["layers"])
                L.exact("WIRING", n_groups == prof.num_layers + 1, f"{n_groups} layer groups", position=j)
                for layer in range(prof.num_layers + 1):
                    g = e["layers"][str(layer)]
                    L.exact(
                        "OPENING_SCHEMA",
                        set(g) in ({"leaf"}, {"tensors"}),
                        f"layer group {layer} must carry either a leaf or tensors, got {sorted(g)}",
                        layer,
                        j,
                    )
                    if "tensors" not in g:
                        continue
                    names, want = set(g["tensors"]), set(prof.group_names(layer))
                    kind = "final" if layer == prof.num_layers else prof.layer_types[layer]
                    L.exact(
                        "WIRING",
                        names == want,
                        f"layer {layer} ({kind}) opened names: missing {sorted(want - names)}, "
                        f"unexpected {sorted(names - want)}",
                        layer,
                        j,
                    )
                    for name, v in g["tensors"].items():
                        if v != "open":
                            continue
                        a = self.tensors.get(tensor_key(j, layer, name))
                        shape = prof.row_shape(layer, name)
                        L.exact(
                            "WIRING",
                            a is not None and a.dtype == np.uint16 and tuple(a.shape) == shape,
                            f"{name} at layer {layer}: got {None if a is None else (a.dtype, a.shape)}, "
                            f"expected bf16 {shape}",
                            layer,
                            j,
                        )
        audited = set(self.positions)
        for j, groups in sorted(req.items()):
            code = "OPENING_SCHEMA" if j in audited else "KV_PROVENANCE"
            with stage(code, j):
                for layer, names in sorted(groups.items()):
                    g = self.entries[j]["layers"].get(str(layer), {})
                    L.exact(code, "tensors" in g, f"layer group {layer} not opened", layer, j)
                    missing = sorted(n for n in names if g["tensors"].get(n) != "open")
                    c = "KV_PROVENANCE" if missing and set(missing) <= {"k_n", "v_n"} else code
                    L.exact(c, not missing, f"required tensors {missing} not opened", layer, j)
        for j in self.opened_decode:
            with stage("OPENING_SCHEMA", j):
                e = self.entries[j]
                pre = self.tensors.get(f"p{j}/logits_precap")
                ok = e.get("logits") == "open" and isinstance(e.get("witness"), dict)
                ok = ok and pre is not None and pre.dtype == np.float32 and pre.shape == (prof.vocab_size,)
                L.exact("OPENING_SCHEMA", ok, "decode position without opened f32 logits and witness", position=j)
        for j in self.positions:
            L.exact(
                "OPENING_SCHEMA",
                f"p{j}/embed_row" in self.tensors and "embedding" in self.entries[j],
                "embedding row not opened",
                position=j,
            )

    def _check_position_leaf(self, j: int, audited: bool) -> None:
        L, prof, r = self.ledger, self.profile, self.receipt
        e = self.entries[j]
        code = "MERKLE_POSITION" if audited or "layers" not in e else "KV_PROVENANCE"
        t = decode_step(r["n_prompt"], r["n_gen"], j)
        lh = None
        with stage(code, j):
            if "layers" not in e:
                body = bytes.fromhex(e["body"])
            else:
                leaves = []
                for layer in range(prof.num_layers + 1):
                    g = e["layers"][str(layer)]
                    if "tensors" not in g:
                        leaves.append(bytes.fromhex(g["leaf"]))
                        continue
                    hs = []
                    for name in prof.group_names(layer):
                        v = g["tensors"][name]
                        hs.append(tensor_hash(self.t(j, layer, name)) if v == "open" else bytes.fromhex(v))
                    leaves.append(layer_leaf(layer, j, hs))
                if any(len(x) != 32 for x in leaves):
                    raise ValueError("layer leaves must be 32-byte digests")
                if t is None:
                    L.exact(
                        code,
                        e["logits"] is None and e["witness"] is None and e["out_token"] is None,
                        "prompt position carries decode data",
                        position=j,
                    )
                    wh = None
                else:
                    lh = (
                        logits_hash(self.tensors[f"p{j}/logits_precap"])
                        if e["logits"] == "open"
                        else bytes.fromhex(e["logits"])
                    )
                    wh = witness_hash(e["witness"]) if isinstance(e["witness"], dict) else bytes.fromhex(e["witness"])
                    if len(lh) != 32 or len(wh) != 32 or not isinstance(e["out_token"], int):
                        raise ValueError("malformed decode fields")
                body = position_body(leaves, lh, e["out_token"], wh)
            if len(body) != 32 or not isinstance(e["input_token"], int) or isinstance(e["input_token"], bool):
                raise ValueError("malformed position entry")
            leaf = position_leaf(j, e["input_token"], body)
            proof = [bytes.fromhex(h) for h in e["proof"]]
            L.exact(
                code,
                verify_proof(leaf, j, self.n_pos, proof, bytes.fromhex(r["trace_root"])),
                f"position {j}: recomputed leaf does not verify against trace_root",
                position=j,
            )
        with stage("IO_CHAIN", j):
            if j < r["n_prompt"]:
                L.exact(
                    "PROMPT_BINDING", e["input_token"] == self.prompt[j], f"input token at {j} != prompt", position=j
                )
            else:
                want = self.transcript[j - r["n_prompt"]][0]
                L.exact("IO_CHAIN", e["input_token"] == want, f"input token at {j} != generated token", position=j)
            if t is not None and lh is not None:
                L.exact(
                    "IO_CHAIN",
                    e["out_token"] == self.transcript[t][0],
                    f"sampled token at {j} != transcript",
                    position=j,
                )
                L.exact("IO_CHAIN", lh == self.transcript[t][1], f"logits hash at {j} != transcript", position=j)

    def _check_audit(self, a: dict[str, Any]) -> None:
        p = a["pos"]
        prof, L = self.profile, self.ledger
        e = self.entries[p]
        with stage("EMBEDDING", p):
            tok = int(e["input_token"])
            row = self.tensors[f"p{p}/embed_row"]
            ok = (
                row.dtype == np.uint16
                and row.shape == (prof.hidden_size,)
                and verify_proof(
                    tensor_hash(row),
                    tok,
                    prof.vocab_size,
                    [bytes.fromhex(h) for h in e["embedding"]["proof"]],
                    bytes.fromhex(self.key.meta["embedding_root"]),
                )
            )
            L.exact("EMBEDDING", ok, f"embedding row of token {tok} does not verify against embedding_root", position=p)
            scale = f32_to_bf16(np.array([prof.embed_scale_bf16], dtype=np.float32))
            bad = int(np.count_nonzero(bf16_mul(row, scale) != self.t(p, 0, "r_in")))
            L.exact("EMBEDDING", bad == 0, f"r_in[0] != bf16(row * {prof.embed_scale_bf16}) at {bad} elements", 0, p)
        for layer in a["layers"]:
            self._check_layer(p, layer, a["attention"])

    def _check_layer(self, p: int, layer: int, with_attention: bool) -> None:
        prof, key, L = self.profile, self.key, self.ledger
        hd, kv, H = prof.head_dims[layer], prof.kv_heads[layer], prof.num_heads
        eps = prof.rms_norm_eps
        T = lambda n: self.t(p, layer, n)  # noqa: E731
        norm = lambda x, y, w: bridge.norm_ratio(x, y, None if w is None else key.norm(layer, w), eps)  # noqa: E731
        for fam, y, x in SHELL_IO:
            if fam in prof.families(layer):
                self.tol(
                    SHELL[fam],
                    lambda fam=fam, y=y, x=x: freivalds.check(
                        key.r(layer, fam), key.v(layer, fam), self.f(p, layer, x), self.f(p, layer, y)
                    ),
                    layer,
                    p,
                )
        self.tol("BRIDGE_NORM_INPUT", lambda: norm(T("r_in"), T("x_attn"), "input_layernorm"), layer, p)
        self.tol("BRIDGE_NORM_Q", lambda: norm(T("q").reshape(H, hd), T("q_n"), "self_attn.q_norm"), layer, p)
        self.tol("BRIDGE_NORM_K", lambda: norm(T("k").reshape(kv, hd), T("k_n"), "self_attn.k_norm"), layer, p)
        if prof.is_global(layer):

            def shared() -> float:
                ref = rmsnorm_gemma(bf16_to_f64(T("k")).reshape(kv, hd), None, eps)
                return float(np.max(bf16_ulp_distance(f32_to_bf16(ref.astype(np.float32)), T("v_n"))))

            self.tol("KV_SHARED", shared, layer, p, unit="ulp")
        else:
            self.tol("BRIDGE_NORM_V", lambda: norm(T("v").reshape(kv, hd), T("v_n"), None), layer, p)
        self.tol("BRIDGE_NORM_POST_ATTN", lambda: norm(T("o"), T("o_n"), "post_attention_layernorm"), layer, p)
        with stage("BRIDGE_RESIDUAL", p, layer):
            bad = bridge.residual_mismatches(T("r_in"), T("o_n"), T("r_mid"))
            L.exact("BRIDGE_RESIDUAL", bad == 0, f"r_mid != bf16(r_in + post_attn) at {bad} elements", layer, p)
        self.tol("BRIDGE_NORM_PRE_FFN", lambda: norm(T("r_mid"), T("x_ffn"), "pre_feedforward_layernorm"), layer, p)
        self.tol("BRIDGE_GELU", lambda: bridge.gelu_ratio(T("g"), T("u"), T("h")), layer, p)
        self.tol("BRIDGE_NORM_POST_FFN", lambda: norm(T("d"), T("d_n"), "post_feedforward_layernorm"), layer, p)
        with stage("BRIDGE_RESIDUAL", p, layer):
            nxt = self.t(p, layer + 1, "r_in") if layer + 1 < prof.num_layers else self.t(p, prof.num_layers, "r_final")
            bad = bridge.residual_mismatches(T("r_mid"), T("d_n"), nxt, key.layer_scalar_bits(layer))
            msg = f"layer output != bf16(bf16(r_mid + post_ffn) * layer_scalar) at {bad} elements"
            L.exact("BRIDGE_RESIDUAL", bad == 0, msg, layer, p)
        if with_attention:

            def replay() -> tuple[float, float]:
                rows = list(prof.window(layer, p))
                k_rows = np.stack([self.t(j, layer, "k_n") for j in rows])
                v_rows = np.stack([self.t(j, layer, "v_n") for j in rows])
                impl = self.receipt["manifest"]["attn_implementation"]
                ref = attention.replay(prof, layer, p, T("q_n"), k_rows, v_rows, rows, impl)
                return attention.deviation(prof, layer, T("a"), ref), attention.ATTN_REL

            self.tol("ATTN_REPLAY", replay, layer, p, unit="rel L2")

    def _check_decode(self, p: int, t: int) -> None:
        prof, key, L, r = self.profile, self.key, self.ledger, self.receipt
        e = self.entries[p]
        n = prof.num_layers
        self.tol(
            "BRIDGE_NORM_FINAL",
            lambda: bridge.norm_ratio(
                self.t(p, n, "r_final"), self.t(p, n, "h_final"), key.final_norm, prof.rms_norm_eps
            ),
            n,
            p,
        )
        pre = self.tensors[f"p{p}/logits_precap"]
        # logits are computed in f32 from bf16 h_final and E: only f32 accumulation error remains
        self.tol(
            "LMHEAD_BINDING",
            lambda: freivalds.check(key.r_lm, key.v_lm, self.f(p, n, "h_final"), pre.astype(np.float64), u_out=F32_U),
            None,
            p,
        )
        with stage("DECODE_SOFTCAP", p):
            post = softcap(pre, prof.final_logit_softcapping)
            L.exact(
                "DECODE_SOFTCAP",
                tensor_hash(post).hex() == e["witness"].get("postcap"),
                "the prover sampled from logits other than softcap(pre-cap logits)",
                position=p,
            )
        with stage("DECODE_SAMPLING", p):
            w = e["witness"]
            token, u = replay_token(post, policy_from_manifest(r["manifest"]), self.seed, t)
            L.exact(
                "DECODE_SAMPLING",
                w.get("step") == t and w.get("u") == u,
                "sampler witness does not match the randomness derived from the revealed seed",
                position=p,
            )
            L.exact(
                "DECODE_SAMPLING",
                token == e["out_token"],
                f"declared policy samples token {token}, committed token is {e['out_token']}",
                position=p,
            )


def spot_check(run: _Run) -> dict[str, Any]:
    """How much of the trace this challenge covers and, if the auditor drew each audit's layers
    uniformly, the probability that a prover forging the same thing at every position escapes it:

    * a residual stream forged at one layer boundary: ``prod_a (1 - |layers_a| / L)`` over all audits;
    * a fake attention output ``a`` at one layer (the audited, not verified, part): the same product
      over the audits that run the attention replay only.
    """
    n_layers = run.profile.num_layers
    escape = attn_escape = 1.0
    for a in run.audits:
        escape *= 1.0 - len(a["layers"]) / n_layers
        if a["attention"]:
            attn_escape *= 1.0 - len(a["layers"]) / n_layers
    return {
        "audits": len(run.audits),
        "layer_audits": sum(len(a["layers"]) for a in run.audits),
        "attention_audits": sum(1 for a in run.audits if a["attention"] and a["layers"]),
        "decode_checked": len(run.opened_decode),
        "n_gen": run.receipt.get("n_gen") if isinstance(run.receipt, dict) else None,
        "forged_boundary_escape": escape,
        "fake_attention_escape": attn_escape,
    }


def coverage(run: _Run, failed: str | None) -> dict[str, str]:
    stats = run.ledger.stats
    prof = run.profile
    sc = spot_check(run)
    out = {}
    for comp, codes in COMPONENTS.items():
        ran = [c for c in codes if c in stats]
        if failed in codes:
            out[comp] = f"FAILED ({failed})"
        elif not ran:
            out[comp] = "not run"
        elif comp == "shell":
            out[comp] = (
                f"verified, tolerance-bounded ({len(ran)}/7 families, {sc['layer_audits']} layer audits at "
                f"{sc['audits']} positions of {prof.num_layers} layers, k={run.key.k})"
            )
        elif comp == "attention":
            out[comp] = (
                f"audited (replay at {sc['attention_audits']} positions, KV provenance, wiring)"
                if "ATTN_REPLAY" in ran
                else "audited (wiring only)"
            )
        elif comp == "bridge":
            out[comp] = "verified at audited layers (norms and GELU tolerance-bounded, residual chain exact)"
        elif comp == "decode":
            out[comp] = (
                f"verified at {sc['decode_checked']}/{sc['n_gen']} generated tokens (soft-cap and sampling exact, "
                f"f32 LM-head binding tolerance-bounded); spot check: escape p={sc['forged_boundary_escape']:.3g} "
                f"for a residual stream forged at one layer boundary, p={sc['fake_attention_escape']:.3g} for a "
                f"fake attention output at one layer"
            )
        elif comp == "embedding":
            out[comp] = "verified (exact)"
        else:
            out[comp] = "verified (every position's input token bound)"
    return out


def verify(
    receipt: dict[str, Any],
    opening: bytes | tuple[dict[str, Any], dict[str, np.ndarray]],
    key: VerifierKey,
    public: dict[str, Any],
    challenge: dict[str, Any],
    prover_id: str | None = None,
    expected: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify ``opening`` against ``receipt`` for the auditor's ``challenge``.

    ``prover_id`` pins the prover's Ed25519 key. ``expected`` holds what the client actually sent
    (``prompt_tokens`` or ``prompt_hash``, ``temperature``, ``top_k``, ``top_p``, ``greedy``,
    ``max_new_tokens``, ``thinking``, ``client_nonce``); without it the receipt only proves
    self-consistency for whatever request the prover claims.
    """
    t0 = time.perf_counter()
    run = _Run(receipt, opening, key, public, challenge, prover_id, expected)
    fail: VerifyFail | None = None
    try:
        run.run()
    except VerifyFail as f:
        fail = f
    ms = (time.perf_counter() - t0) * 1000.0
    verdict: dict[str, Any] = {
        "version": VERDICT_VERSION,
        "result": "PASS" if fail is None else "FAIL",
        "reason": None if fail is None else fail.code,
        "message": None if fail is None else fail.message,
        "layer": None if fail is None else fail.layer,
        "position": None if fail is None else fail.position,
        "detail": {} if fail is None else fail.detail,
        "coverage": coverage(run, None if fail is None else fail.code),
        "checks": run.ledger.summary(),
        "payload_bytes": len(opening) if isinstance(opening, (bytes, bytearray)) else None,
        "verify_ms": round(ms, 3),
        "request_id": receipt.get("request_id") if isinstance(receipt, dict) else None,
        "audits": run.audits,
        "positions": run.positions,
        "layers": sorted({x for a in run.audits for x in a["layers"]}),
        "spot_check": spot_check(run) if run.audits else None,
        "pinned_prover": prover_id is not None,
        "client_expectations": sorted(run.expected),
        "tolerances": {
            "freivalds": f"T=sqrt(mean_j (r_j.y - v_j.x)^2) <= ({freivalds.ROUNDING_MARGIN}*u + "
            f"{freivalds.ACC_MARGIN}*sqrt(K)*2^-24)*||y||_2, u=2^-8 (shell) or 2^-24 (f32 LM head), k={key.k}",
            "norms": "|y-ref| <= 2^-7|ref| + 2^-126 elementwise",
            "gelu": "|h-ref| <= 2^-6|ref| + 2^-20|g*u| + 2^-126 elementwise",
            "kv_shared": "<= 1 bf16 ulp",
            "attention": f"per-head relative L2 <= {attention.ATTN_REL:.4g} (audited)",
        },
    }
    return verdict
