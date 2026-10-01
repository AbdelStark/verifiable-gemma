"""Human-readable rendering of receipts, verdicts, metrics and the claim boundary."""

from __future__ import annotations

from typing import Any

CLAIM_TABLE = [
    ("Embedding row + scale", "verified", "Merkle proof to embedding root; bf16(row * embed_scale) exact"),
    ("Linear shell Wq Wk Wv Wo Wgate Wup Wdown", "verified (tolerance)", "Freivalds, k secret vectors, bf16 bound"),
    ("LM head (f32 logits)", "verified (tolerance)", "Freivalds of the logits against h_final, f32 bound"),
    ("Norms (4 per layer, q/k/v, final)", "verified (tolerance)", "float64 replay, elementwise 2^-7 relative"),
    ("GELU-tanh gate", "verified (tolerance)", "float64 replay, elementwise 2^-6 relative"),
    ("Residual chain + layer_scalar", "verified (exact)", "bf16 round-to-nearest-even replay"),
    ("Shared K/V on global layers", "verified (1 ulp)", "v_n == bf16(rmsnorm(k_pre))"),
    ("Soft-cap and sampled token, every token", "verified (exact)", "canonical soft-cap, shared sampler, nonce seed"),
    ("Bindings: prompt, request, manifest, seed, count, IO", "verified (exact)", "hashes, signature, Merkle"),
    ("K/V provenance", "audited", "every attended row Merkle-bound to the trace root"),
    ("Attention output at challenged positions", "audited", "single-query float64 replay, tolerance 2^-5"),
    ("Wiring: layer types, window, head dims, RoPE, QK-norm", "audited", "manifest hashes + opened shapes"),
    ("Positions and layers not drawn by the auditor", "open", "spot check: random positions, routine layers"),
    ("Consistent fake attention output a", "open", "CommitLLM residual hole; needs Q retention or fixed kernels"),
    ("Shell deviations below the bf16 bound", "open", "~1.2% of an output's L2 norm, maybe on few elements; Tier 2"),
]


def short(h: str, n: int = 12) -> str:
    return h[:n] + "..." if isinstance(h, str) and len(h) > n else str(h)


def fmt_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024 or unit == "GiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} GiB"


def format_receipt(r: dict[str, Any]) -> str:
    m = r["manifest"]
    lines = [
        f"receipt {r['request_id']}  (version {r['version']})",
        f"  model         {r['model']['id']}  revision {short(r['model']['revision'], 16)}",
        f"  weights_root  {short(r['model']['weights_root'], 24)}  config_hash {short(r['model']['config_hash'], 16)}",
        f"  manifest      {m['dtype']} {m['attn_implementation']}  T={m['temperature']} top_k={m['top_k']} "
        f"top_p={m['top_p']} greedy={m['greedy']}  softcap={m['final_logit_softcapping']} "
        f"embed_scale={m['embed_scale_bf16']}  thinking={m['thinking']}",
        f"  tokens        n_prompt={r['n_prompt']} n_gen={r['n_gen']}",
        f"  trace_root    {short(r['trace_root'], 24)}  io_chain_head {short(r['io_chain_head'], 16)}",
        f"  prover        {short(r['prover']['id'], 24)}  signed",
    ]
    return "\n".join(lines)


def format_verdict(v: dict[str, Any], deep: bool = False) -> str:
    head = f"VERDICT {v['result']}"
    if v["result"] == "FAIL":
        loc = []
        if v.get("layer") is not None:
            loc.append(f"layer {v['layer']}")
        if v.get("position") is not None:
            loc.append(f"position {v['position']}")
        head += f"  reason {v['reason']}" + (f" ({', '.join(loc)})" if loc else "")
        head += f"\n  {v['message']}"
    lines = [head]
    pb = v.get("payload_bytes")
    lines.append(
        f"  positions {v['positions']}  layers {len(v['layers'])}  opening {fmt_bytes(pb) if pb else '-'}  "
        f"verify {v['verify_ms']:.1f} ms"
    )
    lines.append("  coverage:")
    for k, s in v["coverage"].items():
        lines.append(f"    {k:10s} {s}")
    rows = [(c, s) for c, s in v["checks"].items() if deep or s["kind"] != "exact"]
    if rows:
        lines.append("  checks (deviation <= bound):" if not deep else "  all checks:")
        for c, s in rows:
            w = s.get("worst")
            if w:
                unit = f" {w['unit']}" if w.get("unit") else ""
                worst = f"worst {w['deviation']:.3e} <= {w['tolerance']:.3e}{unit}"
                lines.append(
                    f"    {c:22s} {s['kind']:9s} n={s['n']:<5d} {worst}  ({100 * s['max_ratio']:.1f}% of bound)"
                )
            else:
                lines.append(f"    {c:22s} {s['kind']:9s} n={s['n']:<5d}")
    return "\n".join(lines)


def format_overhead(ov: dict[str, Any]) -> list[str]:
    p, c = ov["plain"], ov["capture"]
    return [
        f"  decode tokens/s     plain {p['decode_tok_s']:.1f}  capture {c['decode_tok_s']:.1f}  "
        f"overhead {100 * ov['decode_overhead']:.1f}%",
        f"  serve tokens/s      plain {p['serve_tok_s']:.1f}  capture+commit {c['serve_tok_s']:.1f}  "
        f"overhead {100 * ov['serve_overhead']:.1f}%",
    ]


def format_claims() -> str:
    w = max(len(a) for a, _, _ in CLAIM_TABLE)
    lines = ["claim boundary (CommitLLM vocabulary):"]
    for comp, status, how in CLAIM_TABLE:
        lines.append(f"  {comp:{w}s}  {status:21s} {how}")
    return "\n".join(lines)
