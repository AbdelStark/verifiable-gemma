"""``vg demo`` and ``vg bench``.

The demo starts the provider as a separate process per scenario (honest, then each tamper mode)
that is given only ``public.json`` and a state directory; this process keeps the secret key and
plays the auditor and verifier over HTTP.
"""

from __future__ import annotations

import json
import secrets
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vgemma.auditor import (
    ProverClient,
    challenge_from,
    client_request,
    expected_from_request,
    forged_boundary_escape,
    make_challenge,
    routine_k,
)
from vgemma.display import fmt_bytes, format_claims, format_overhead, format_receipt, format_verdict
from vgemma.prover.tamper import DEMO_MODES, EXPECTED_CODES
from vgemma.verifier.key import VerifierKey
from vgemma.verifier.verify import verify

DEFAULT_PROMPT = "Write two sentences about why open model weights make audits possible."


@dataclass
class Setup:
    model: str
    key_path: Path
    public_path: Path
    state_dir: Path
    keygen_s: float | None


def prepare(
    model: str | None,
    tiny: bool,
    workdir: Path,
    log=print,
    key_dir: Path | None = None,
    public_path: Path | None = None,
) -> Setup:
    """Build the tiny checkpoint and run keygen if needed. Keys default to ``workdir/keys/<name>``
    and public params to ``workdir/public/<name>``; Modal passes ``/keys/...`` and ``/state/...``."""
    from vgemma.keygen import keygen
    from vgemma.tiny import build_tiny

    workdir.mkdir(parents=True, exist_ok=True)
    if tiny:
        model_dir = workdir / "tiny-gemma4"
        if not (model_dir / "config.json").exists():
            log(f"building tiny Gemma 4 checkpoint -> {model_dir}")
            build_tiny(model_dir)
        model = str(model_dir)
    if model is None:
        raise ValueError("--model is required unless --tiny")
    name = "tiny" if tiny else model.replace("/", "--")
    key_dir = Path(key_dir) if key_dir else workdir / "keys" / name
    public_path = Path(public_path) if public_path else workdir / "public" / name / "public.json"
    key_path = key_dir / "key.npz"
    keygen_s = None
    if not (key_path.exists() and public_path.exists()):
        log(f"keygen for {model}")
        pub = keygen(model, key_dir, public_out=public_path, log=lambda s: log("  " + s))
        keygen_s = pub["keygen"]["seconds"]
    return Setup(model, key_path, public_path, workdir / "state", keygen_s)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ServerProcess:
    """`vg serve` in a child process; it only receives the public params and a state directory."""

    def __init__(self, setup: Setup, device: str, tamper: str | None, log_path: Path, startup_timeout: float):
        self.port = _free_port()
        cmd = [
            sys.executable,
            "-m",
            "vgemma.cli",
            "serve",
            "--model",
            setup.model,
            "--public",
            str(setup.public_path),
            "--state-dir",
            str(setup.state_dir),
            "--port",
            str(self.port),
            "--device",
            device,
        ]
        if tamper:
            cmd += ["--tamper", tamper]
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path = log_path
        self.log_file = open(log_path, "w")  # noqa: SIM115
        self.proc = subprocess.Popen(cmd, stdout=self.log_file, stderr=subprocess.STDOUT)
        self.client = ProverClient(f"http://127.0.0.1:{self.port}")
        deadline = time.time() + startup_timeout
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited during startup, see {log_path}:\n{log_path.read_text()[-3000:]}")
            try:
                self.health = self.client.health()
                break
            except Exception:  # noqa: BLE001
                if time.time() > deadline:
                    self.stop()
                    raise RuntimeError(f"server did not become healthy, see {log_path}") from None
                time.sleep(0.5)

    def tamper_banner(self) -> list[str]:
        return [
            ln.rstrip() for ln in self.log_path.read_text().splitlines() if "TAMPER" in ln or ln.startswith("tamper ")
        ]

    def stop(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log_file.close()


def run_demo(
    model: str | None,
    tiny: bool,
    device: str,
    workdir: Path,
    prompt: str = DEFAULT_PROMPT,
    max_new_tokens: int = 24,
    startup_timeout: float = 1800.0,
    tamper_layers: str = "full",
    key_dir: Path | None = None,
    public_path: Path | None = None,
    log=print,
) -> dict[str, Any]:
    from vgemma.model import resolve_tokenizer_dir

    setup = prepare(model, tiny, workdir, log, key_dir, public_path)
    key = VerifierKey(setup.key_path)
    public = json.loads(setup.public_path.read_text())
    n_layers = key.profile.num_layers
    tokenizer_dir = resolve_tokenizer_dir(setup.model)  # the client templates its own prompt
    runs_dir = workdir / "runs"
    results: list[dict[str, Any]] = []
    prover_id = None
    costs: dict[str, Any] = {}

    def section(title: str) -> None:
        log("")
        log("=" * 78)
        log(title)
        log("=" * 78)

    scenarios: list[tuple[str, str | None]] = [("honest", None)] + [(m, m) for m in DEMO_MODES]
    for name, mode in scenarios:
        section(f"[{name}] provider {'with --tamper ' + mode if mode else 'honest'}")
        srv = ServerProcess(setup, device, mode, workdir / "logs" / f"serve-{name}.log", startup_timeout)
        try:
            for line in srv.tamper_banner():
                log(f"  server: {line}")
            if prover_id is None:
                prover_id = srv.health["prover_id"]  # pinned on first contact
                log(f"  prover id pinned: {prover_id[:32]}...")
            req = client_request(
                prompt, max_new_tokens, tokenizer_dir, prover_id=prover_id,
                attn_implementation=srv.health.get("attn_implementation"),
            )  # fmt: skip
            resp = srv.client.chat(prompt, **req["params"])
            receipt = resp["receipt"]
            out_dir = runs_dir / name
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "receipt.json").write_text(json.dumps(receipt, indent=1))
            (out_dir / "request.json").write_text(json.dumps(req, indent=1))
            note = "  (random tiny weights: the text is noise)" if tiny else ""
            log(f"  response: {resp['text']!r}{note}")
            log("  " + format_receipt(receipt).replace("\n", "\n  "))
            log("  nothing is proved yet: the receipt only commits to the trace.")
            audits = [("routine", "routine"), ("full", "full")] if mode is None else [(tamper_layers, tamper_layers)]
            for label, layers in audits:
                ch = make_challenge(receipt, n_layers, positions="random", layers=layers)
                t0 = time.perf_counter()
                opening = srv.client.audit(ch)
                audit_s = time.perf_counter() - t0
                (out_dir / f"opening-{label}.bin").write_bytes(opening)
                (out_dir / f"challenge-{label}.json").write_text(json.dumps(ch, indent=1))
                verdict = verify(
                    receipt,
                    opening,
                    key,
                    public,
                    challenge=ch,
                    prover_id=prover_id,
                    expected=expected_from_request(req),
                )
                (out_dir / f"verdict-{label}.json").write_text(json.dumps(verdict, indent=1, default=str))
                full = [(a["pos"], a["layers"]) for a in ch["audits"] if a["attention"]]
                escape = forged_boundary_escape(ch, n_layers)
                attn_escape = forged_boundary_escape(ch, n_layers, attention_only=True)
                log(f"\n  audit [{ch['tier']}] full audits at random positions {full}")
                log(
                    f"  + every generated token decode-checked with one random layer audited; a residual "
                    f"stream forged at one boundary escapes with p={escape:.3g}, a fake attention output "
                    f"at one layer with p={attn_escape:.3g}"
                )
                log(f"  opening served in {audit_s:.2f} s")
                log("  " + format_verdict(verdict).replace("\n", "\n  "))
                expected = "PASS" if mode is None else EXPECTED_CODES[mode]
                got = verdict["result"] if mode is None else verdict["reason"]
                results.append(
                    {
                        "scenario": name,
                        "audit": ch["tier"],
                        "result": verdict["result"],
                        "reason": verdict["reason"],
                        "expected": expected,
                        "ok": got == expected,
                        "layer": verdict["layer"],
                        "detail": verdict["detail"],
                    }
                )
                if mode is None:
                    costs[label] = {
                        "opening_bytes": len(opening),
                        "verify_ms": verdict["verify_ms"],
                        "audits": len(ch["audits"]),
                        "forged_boundary_escape": escape,
                        "fake_attention_escape": attn_escape,
                    }
            if mode is None:
                t = resp["timings"]
                n_pos = receipt["n_prompt"] + receipt["n_gen"] - 1
                costs.update(
                    retained_bytes_per_position=t["retained_bytes"] / n_pos, n_positions=n_pos, commit_s=t["commit_s"]
                )
                costs["overhead"] = srv.client.bench(prompt, max_new_tokens, runs=3)
        finally:
            srv.stop()

    section("summary")
    for r in results:
        mark = "ok " if r["ok"] else "MISMATCH"
        extra = ""
        if r["detail"].get("deviation") is not None:
            extra = f"  deviation {r['detail']['deviation']:.3g} > tolerance {r['detail']['tolerance']:.3g}"
        if r["layer"] is not None:
            extra = f"  layer {r['layer']}" + extra
        log(
            f"  [{mark}] {r['scenario']:9s} {r['audit']:10s} {r['result']:4s} {r['reason'] or '':16s} "
            f"(expected {r['expected']}){extra}"
        )
    ov = costs.get("overhead", {})
    log("")
    log("cost summary:")
    for line in format_overhead(ov) if ov else []:
        log(line)
    log(
        f"  retained state        {fmt_bytes(costs['retained_bytes_per_position'])} per position "
        f"({costs['n_positions']} positions)"
    )
    for label in ("routine", "full"):
        c = costs[label]
        log(
            f"  audit {label:8s}        opening {fmt_bytes(c['opening_bytes'])} for {c['audits']} audits, "
            f"verify {c['verify_ms']:.1f} ms, escape p={c['forged_boundary_escape']:.3g} (residual) "
            f"/ {c['fake_attention_escape']:.3g} (attention)"
        )
    kg = public.get("keygen", {})
    log(
        f"  keygen                {kg.get('seconds', float('nan')):.2f} s, key {fmt_bytes(kg.get('key_bytes', 0))} "
        f"(k={kg.get('freivalds_k')})"
    )
    log("")
    log(format_claims())
    ok = all(r["ok"] for r in results)
    log("")
    log("DEMO " + ("PASSED: honest PASS, every tamper rejected with the expected reason" if ok else "FAILED"))
    summary = {"ok": ok, "results": results, "costs": costs, "keygen": kg}
    (workdir / "demo-summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary


def run_bench(
    model: str | None,
    tiny: bool,
    device: str,
    workdir: Path,
    prompt: str = DEFAULT_PROMPT,
    max_new_tokens: int = 32,
    runs: int = 3,
    calibrate: bool = False,
    key_dir: Path | None = None,
    public_path: Path | None = None,
    log=print,
) -> dict[str, Any]:
    """In-process measurements: capture overhead, retained bytes, opening bytes, verify ms, and the
    honest deviation as a fraction of each tolerance bound (``calibrate``: every generated position)."""
    from vgemma.model import load_model
    from vgemma.prover.engine import Engine, ProverIdentity, measure_overhead
    from vgemma.prover.store import RetainedStore
    from vgemma.tokenizer import load_tokenizer

    setup = prepare(model, tiny, workdir, log, key_dir, public_path)
    public = json.loads(setup.public_path.read_text())
    key = VerifierKey(setup.key_path)
    lm = load_model(setup.model, device=device)
    tok = load_tokenizer(lm.model_dir, lm.eos_token_ids)
    store = RetainedStore(setup.state_dir / "retained")
    engine = Engine(
        lm, tok, store, ProverIdentity.load_or_create(setup.state_dir / "prover_identity.json"), public, log=log
    )
    msgs = [{"role": "user", "content": prompt}]
    engine.generate(messages=msgs, max_new_tokens=4, retain=False)  # warm-up
    ov = measure_overhead(engine, msgs, max_new_tokens, runs)
    res = engine.generate(messages=msgs, max_new_tokens=max_new_tokens)
    r = res.receipt
    n_pos = r["n_prompt"] + r["n_gen"] - 1
    n_layers = key.profile.num_layers
    rows: dict[str, Any] = {}
    margins: dict[str, float] = {}
    rng = secrets.SystemRandom()
    gen = list(range(r["n_prompt"] - 1, n_pos))
    three = sorted(rng.sample(gen, min(3, len(gen))))
    routine = sorted(rng.sample(range(n_layers), routine_k(n_layers)))
    audits = [
        ("routine", three, routine, True),
        ("routine-noattn", three, routine, False),
        ("full", three, list(range(n_layers)), True),
        ("full-noattn", three, list(range(n_layers)), False),
        ("one-layer-noattn", three, [rng.randrange(n_layers)], False),
    ]
    if calibrate:
        audits.append(("calibrate", gen, list(range(n_layers)), True))
    for label, positions, layers, attn in audits:
        ch = challenge_from(r["request_id"], positions, layers, attention=attn)
        opening = engine.open(r["request_id"], ch)
        times = []
        for _ in range(runs):
            v = verify(r, opening, key, public, challenge=ch)
            times.append(v["verify_ms"])
            if v["result"] != "PASS":
                raise RuntimeError(f"bench audit failed: {v['reason']} {v['message']}")
        for code, st in v["checks"].items():
            if st["worst"] is not None:
                margins[code] = max(margins.get(code, 0.0), st["max_ratio"])
        n = len(positions)
        rows[label] = {
            "layers": len(layers),
            "positions": n,
            "opening_bytes_per_position": len(opening) / n,
            "verify_ms_per_position": sorted(times)[len(times) // 2] / n,
        }
    default = make_challenge(r, n_layers)
    op = engine.open(r["request_id"], default)
    v = verify(r, op, key, public, challenge=default)
    if v["result"] != "PASS":
        raise RuntimeError(f"bench default audit failed: {v['reason']} {v['message']}")
    rows["default-challenge"] = {
        "layers": routine_k(n_layers),
        "positions": len(default["audits"]),
        "opening_bytes": len(op),
        "verify_ms": v["verify_ms"],
        "forged_boundary_escape": forged_boundary_escape(default, n_layers),
        "opening_bytes_per_position": len(op) / len(default["audits"]),
        "verify_ms_per_position": v["verify_ms"] / len(default["audits"]),
    }
    kg = public.get("keygen", {})
    metrics = {
        "model": setup.model,
        "device": device,
        "n_prompt": r["n_prompt"],
        "n_gen": r["n_gen"],
        "overhead": ov,
        "retained_bytes_per_position": res.timings["retained_bytes"] / n_pos,
        "commit_s": res.timings["commit_s"],
        "audits": rows,
        "honest_deviation_over_bound": margins,
        "keygen_s": kg.get("seconds"),
        "key_bytes": kg.get("key_bytes"),
        "freivalds_k": kg.get("freivalds_k"),
    }
    log(f"bench: {setup.model} on {device}, prompt {r['n_prompt']} tokens, {r['n_gen']} generated")
    for line in format_overhead(ov):
        log(line)
    log(f"  retained state     {fmt_bytes(metrics['retained_bytes_per_position'])} per position")
    for k, row in rows.items():
        size, ms = fmt_bytes(row["opening_bytes_per_position"]), row["verify_ms_per_position"]
        log(f"  audit {k:14s} {row['layers']:3d} layers  opening {size}/position  verify {ms:.1f} ms/position")
    log(f"  keygen             {kg.get('seconds', float('nan')):.2f} s, key {fmt_bytes(kg.get('key_bytes', 0))}")
    log("  honest deviation as a fraction of each bound (worst over all audited checks):")
    for code, m in sorted(margins.items(), key=lambda kv: -kv[1]):
        log(f"    {code:22s} {100 * m:6.1f}%")
    (workdir / "bench.json").write_text(json.dumps(metrics, indent=1, default=str))
    return metrics
