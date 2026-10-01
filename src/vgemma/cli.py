"""``vg`` command line (TECH_SPEC section 13)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True, help="verifiable-gemma: commit-and-audit for Gemma 4")


@app.command()
def tiny(out: Path = typer.Option(Path("tiny-gemma4"), help="output checkpoint directory"), seed: int = 0) -> None:
    """Build the tiny random Gemma 4 checkpoint used for CPU tests."""
    from vgemma.tiny import build_tiny

    d = build_tiny(out, seed=seed)
    typer.echo(f"tiny Gemma 4 checkpoint written to {d}")


@app.command()
def keygen(
    model: str = typer.Option(..., help="checkpoint directory or Hub id"),
    out: Path = typer.Option(..., help="directory for key.npz (secret)"),
    public_out: Path | None = typer.Option(None, help="public.json path (default: <out>/public.json)"),
    k: int = typer.Option(16, help="Freivalds vectors per matrix family"),
) -> None:
    """Build the secret verifier key and the public params from a public checkpoint."""
    from vgemma.keygen import keygen as run

    run(model, out, k=k, public_out=public_out, log=typer.echo)


@app.command()
def serve(
    model: str = typer.Option(..., help="checkpoint directory or Hub id"),
    public: Path = typer.Option(..., help="public.json from keygen (the server never reads key.npz)"),
    port: int = 8000,
    host: str = "127.0.0.1",
    device: str = typer.Option("cpu", help="cpu | cuda"),
    state_dir: Path = typer.Option(Path("state"), help="retained state, prover identity"),
    tamper: str | None = typer.Option(None, help="DEMO ONLY: weights | identity | sampling | softcap | ..."),
    attn: str = typer.Option("sdpa", help="sdpa | eager"),
    ttl: float = typer.Option(3600.0, help="retained-state TTL in seconds"),
    max_requests: int = typer.Option(64, help="maximum retained requests"),
) -> None:
    """Run the provider with capture hooks: /chat, /audit, /health."""
    import uvicorn

    from vgemma.model import load_model
    from vgemma.prover.engine import Engine, ProverIdentity
    from vgemma.prover.server import create_app
    from vgemma.prover.store import RetainedStore
    from vgemma.prover.tamper import Tamper
    from vgemma.tokenizer import load_tokenizer

    if public.name == "key.npz" or public.suffix == ".npz":
        raise typer.BadParameter("pass public.json, never the secret key")
    pub = json.loads(public.read_text())
    lm = load_model(model, device=device, attn_implementation=attn)
    tok = load_tokenizer(lm.model_dir, lm.eos_token_ids)
    store = RetainedStore(state_dir / "retained", ttl_seconds=ttl, max_requests=max_requests)
    identity = ProverIdentity.load_or_create(state_dir / "prover_identity.json")
    engine = Engine(lm, tok, store, identity, public=pub, tamper=Tamper(tamper) if tamper else None, log=typer.echo)
    typer.echo(f"serving {lm.model_id} on {host}:{port} ({device}, {attn}); prover {identity.id[:24]}...")
    uvicorn.run(create_app(engine), host=host, port=port, log_level="warning")


@app.command()
def chat(
    server: str = typer.Option("http://127.0.0.1:8000"),
    prompt: str = typer.Option(...),
    max_new_tokens: int = 128,
    thinking: bool = False,
    greedy: bool = False,
    temperature: float = 1.0,
    top_k: int = 64,
    top_p: float = 0.95,
    model: str = typer.Option(..., help="checkpoint dir or Hub id: its public tokenizer templates the prompt"),
    out: Path = typer.Option(Path("receipt.json"), help="where to save the receipt"),
    request_out: Path = typer.Option(Path("request.json"), help="what was asked, for vg verify --request"),
) -> None:
    """Send a prompt with a fresh client nonce; print the answer, save the receipt and the request."""
    from vgemma.auditor import ProverClient, client_request
    from vgemma.display import format_receipt
    from vgemma.model import resolve_tokenizer_dir

    client = ProverClient(server)
    health = client.health()
    req = client_request(
        prompt, max_new_tokens, resolve_tokenizer_dir(model), thinking=thinking, greedy=greedy,
        temperature=temperature, top_k=top_k, top_p=top_p, prover_id=health["prover_id"],
        attn_implementation=health.get("attn_implementation"),
    )  # fmt: skip
    resp = client.chat(prompt, **req["params"])
    out.write_text(json.dumps(resp["receipt"], indent=1))
    request_out.write_text(json.dumps(req, indent=1))
    typer.echo(resp["text"])
    typer.echo("")
    typer.echo(format_receipt(resp["receipt"]))
    typer.echo(f"receipt saved to {out}, request to {request_out}")


@app.command()
def audit(
    server: str = typer.Option("http://127.0.0.1:8000"),
    receipt: Path = typer.Option(Path("receipt.json")),
    positions: str = typer.Option("random", help="full audits: random | random:k | all-gen | gen:i,... | 17,98"),
    layers: str = typer.Option("routine", help="per full audit: routine | routine:k | full | 0,5,11"),
    decode: str = typer.Option("all-gen", help="decode audits: all-gen | random:k | none"),
    decode_layers: int = typer.Option(1, help="random layers audited in full at each decode audit"),
    decode_attention: bool = typer.Option(False, help="also replay attention at decode audits (opens K/V rows)"),
    no_attention: bool = typer.Option(False, help="skip KV rows and attention replay"),
    out: Path = typer.Option(Path("opening.bin")),
    challenge_out: Path = typer.Option(Path("challenge.json")),
) -> None:
    """Challenge a committed response with the auditor's randomness; save the challenge and the opening."""
    from vgemma.auditor import ProverClient, forged_boundary_escape, make_challenge
    from vgemma.display import fmt_bytes

    rec = json.loads(receipt.read_text())
    client = ProverClient(server)
    n_layers = json.loads(client.health()["profile"])["layers"]
    ch = make_challenge(
        rec, n_layers, positions=positions, layers=layers, attention=not no_attention, decode=decode,
        decode_layers=decode_layers, decode_attention=decode_attention,
    )  # fmt: skip
    data = client.audit(ch)
    out.write_bytes(data)
    challenge_out.write_text(json.dumps(ch, indent=1))
    full = [(a["pos"], a["layers"]) for a in ch["audits"] if a["attention"]]
    typer.echo(
        f"full audits {full}; {len(ch['audits']) - len(full)} decode audits, {decode_layers} random layer(s) each"
    )
    escape, attn = forged_boundary_escape(ch, n_layers), forged_boundary_escape(ch, n_layers, attention_only=True)
    typer.echo(
        f"spot check: escape p={escape:.3g} for a residual stream forged at one layer boundary, "
        f"p={attn:.3g} for a fake attention output at one layer"
    )
    typer.echo(f"opening {fmt_bytes(len(data))} saved to {out}, challenge to {challenge_out}")


@app.command()
def verify(
    key: Path = typer.Option(..., help="key.npz (secret)"),
    public: Path = typer.Option(..., help="public.json"),
    receipt: Path = typer.Option(Path("receipt.json")),
    opening: Path = typer.Option(Path("opening.bin")),
    challenge: Path = typer.Option(Path("challenge.json"), help="the auditor's challenge from vg audit"),
    request: Path | None = typer.Option(None, help="request.json from vg chat: binds prompt, policy, nonce, prover"),
    prover_id: str | None = typer.Option(None, help="pin the prover's ed25519 id (default: from --request)"),
    deep: bool = typer.Option(False, help="print every check"),
    out: Path | None = typer.Option(None, help="write the verdict JSON here"),
) -> None:
    """Verify an opening against a receipt on CPU, without the model weights."""
    from vgemma.auditor import expected_from_request
    from vgemma.display import format_verdict
    from vgemma.verifier.key import VerifierKey
    from vgemma.verifier.verify import verify as run

    expected, pinned = {}, prover_id
    if request is not None:
        req = json.loads(request.read_text())
        expected = expected_from_request(req)
        pinned = pinned or req.get("prover_id")
    if pinned is None:
        typer.echo("warning: prover id not pinned; the signature only proves the receipt is self-signed")
    if request is None:
        typer.echo("warning: no --request; the prompt and sampling policy are only checked for self-consistency")
    v = run(
        json.loads(receipt.read_text()),
        opening.read_bytes(),
        VerifierKey(key),
        json.loads(public.read_text()),
        challenge=json.loads(challenge.read_text()),
        prover_id=pinned,
        expected=expected,
    )
    typer.echo(format_verdict(v, deep=deep))
    if out:
        out.write_text(json.dumps(v, indent=1, default=str))
    raise typer.Exit(0 if v["result"] == "PASS" else 1)


@app.command()
def demo(
    model: str | None = typer.Option(None, help="checkpoint directory or Hub id (default with --tiny: built)"),
    tiny: bool = typer.Option(False, "--tiny", help="run on the tiny CPU checkpoint"),
    device: str = "cpu",
    workdir: Path = typer.Option(Path(".vg"), help="checkpoint, keys, state, logs, runs"),
    max_new_tokens: int = 24,
    prompt: str | None = None,
    tamper_layers: str = typer.Option("full", help="layer set for the tamper audits"),
) -> None:
    """Honest PASS, four tamper FAILs with reason codes, cost summary."""
    from vgemma.demo import DEFAULT_PROMPT, run_demo

    s = run_demo(
        model,
        tiny,
        device,
        workdir,
        prompt=prompt or DEFAULT_PROMPT,
        max_new_tokens=max_new_tokens,
        tamper_layers=tamper_layers,
        log=typer.echo,
    )
    raise typer.Exit(0 if s["ok"] else 1)


@app.command()
def bench(
    model: str | None = typer.Option(None),
    tiny: bool = typer.Option(False, "--tiny"),
    device: str = "cpu",
    workdir: Path = typer.Option(Path(".vg")),
    max_new_tokens: int = 32,
    runs: int = 3,
    calibrate: bool = typer.Option(False, help="also audit every generated position on all layers"),
) -> None:
    """Capture overhead, retained bytes, opening bytes, verifier time, honest deviation margins."""
    from vgemma.demo import run_bench

    run_bench(
        model, tiny, device, workdir, max_new_tokens=max_new_tokens, runs=runs, calibrate=calibrate, log=typer.echo
    )


if __name__ == "__main__":
    sys.exit(app())
