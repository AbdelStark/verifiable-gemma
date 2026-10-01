"""Provider HTTP API: ``POST /chat``, ``POST /audit``, ``GET /health`` (plus ``POST /bench``).

The server reads only ``public.json``; it never sees the verifier key.
"""

from __future__ import annotations

import threading
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from vgemma.prover.engine import Engine, measure_overhead
from vgemma.prover.sampler import SamplingPolicy


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message] | None = None
    prompt: str | None = None
    max_new_tokens: int = Field(64, ge=1, le=8192)
    thinking: bool = False
    temperature: float = 1.0
    top_k: int = 64
    top_p: float = 0.95
    greedy: bool = False
    nonce: str | None = Field(None, pattern="^[0-9a-f]{64}$")


class Audit(BaseModel):
    pos: int
    layers: list[int]
    attention: bool = False


class AuditRequest(BaseModel):
    request_id: str
    audits: list[Audit]
    tier: str = "custom"


class BenchRequest(BaseModel):
    prompt: str = "Explain in one paragraph why the sky is blue."
    max_new_tokens: int = Field(32, ge=1, le=4096)
    runs: int = Field(3, ge=1, le=20)


def _messages(req: ChatRequest) -> list[dict[str, str]]:
    if req.messages:
        return [m.model_dump() for m in req.messages]
    if req.prompt is not None:
        return [{"role": "user", "content": req.prompt}]
    raise HTTPException(400, "messages or prompt required")


def create_app(engine: Engine) -> FastAPI:
    app = FastAPI(title="verifiable-gemma prover")
    lock = threading.Lock()  # one model, one request at a time

    @app.get("/health")
    def health() -> dict[str, Any]:
        return engine.health()

    @app.post("/chat")
    def chat(req: ChatRequest) -> dict[str, Any]:
        policy = SamplingPolicy(temperature=req.temperature, top_k=req.top_k, top_p=req.top_p, greedy=req.greedy)
        with lock:
            res = engine.generate(
                messages=_messages(req),
                max_new_tokens=req.max_new_tokens,
                policy=policy,
                thinking=req.thinking,
                client_nonce=bytes.fromhex(req.nonce) if req.nonce else None,
            )
        return {"text": res.text, "token_ids": res.token_ids, "receipt": res.receipt, "timings": res.timings}

    @app.post("/audit")
    def audit(req: AuditRequest) -> Response:
        with lock:
            try:
                data = engine.open(req.request_id, req.model_dump())
            except KeyError as e:
                raise HTTPException(404, str(e)) from e
            except ValueError as e:
                raise HTTPException(400, str(e)) from e
        return Response(content=data, media_type="application/octet-stream")

    @app.post("/bench")
    def bench(req: BenchRequest) -> dict[str, Any]:
        """Serving overhead: the same request with capture off and on (nothing is retained)."""
        with lock:
            return measure_overhead(engine, [{"role": "user", "content": req.prompt}], req.max_new_tokens, req.runs)

    return app
