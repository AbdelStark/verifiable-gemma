"""Test helpers: a committed run with its opening, opening and receipt mutation, malicious recommits."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable
from typing import Any

from vgemma.auditor import challenge_from, choose_layers, choose_positions, make_challenge
from vgemma.protocol import decode_opening, encode_opening, sign_receipt
from vgemma.prover.engine import Engine, build_opening, build_receipt, commit_trace
from vgemma.prover.sampler import SamplingPolicy
from vgemma.verifier.verify import verify

PROMPT = "Hello tiny Gemma, please tell me a short story about a lighthouse."


# ---------------------------------------------------------------------------


class Run:
    """One committed request plus a challenge and its opening.

    By default the challenge is fixed for reproducible tests: full audits (``layers``, with
    attention) at ``positions``, and decode audits at ``decode`` positions auditing
    ``decode_layers`` (none by default). ``random=True`` uses the auditor's production challenge.
    """

    def __init__(
        self,
        engine: Engine,
        max_new_tokens: int = 12,
        positions: str = "edges",
        layers: str = "full",
        policy: SamplingPolicy | None = None,
        request_id: str | None = None,
        attention: bool = True,
        prompt: str = PROMPT,
        decode: str = "all-gen",
        nonce: bytes | None = None,
        decode_layers: tuple[int, ...] = (),
        random: bool = False,
    ):
        self.engine = engine
        if nonce is None and request_id is not None:  # reproducible across test sessions
            nonce = hashlib.sha256(b"test-nonce/" + request_id.encode()).digest()
        self.result = engine.generate(
            messages=[{"role": "user", "content": prompt}],
            max_new_tokens=max_new_tokens,
            policy=policy,
            request_id=request_id,
            client_nonce=nonce,
        )
        self.receipt = self.result.receipt
        n_layers = engine.profile.num_layers
        if random:
            self.challenge = make_challenge(self.receipt, n_layers, positions=positions, layers=layers)
            self.full_positions = [a["pos"] for a in self.challenge["audits"] if a["attention"]]
            self.layers, self.attention, self.decode_positions, self.decode_layers = None, True, [], ()
        else:
            self.full_positions = choose_positions(positions, self.receipt)
            self.layers, _ = choose_layers(layers, n_layers)
            self.attention = attention
            self.decode_positions = sorted(set(choose_positions(decode, self.receipt)) - set(self.full_positions))
            self.decode_layers = tuple(decode_layers)
            self.challenge = self.with_()
        self.opening = engine.open(self.receipt["request_id"], self.challenge)

    def with_(self, positions=None, layers=None, attention=None, decode_positions=None, decode_layers=None) -> dict:
        """This run's fixed challenge with some fields replaced."""
        return challenge_from(
            self.receipt["request_id"],
            self.full_positions if positions is None else positions,
            self.layers if layers is None else layers,
            attention=self.attention if attention is None else attention,
            decode_positions=self.decode_positions if decode_positions is None else decode_positions,
            decode_layers=self.decode_layers if decode_layers is None else decode_layers,
        )

    @property
    def trace(self):
        return self.result.trace


def check(keys, receipt, opening, challenge, prover_id=None, expected=None) -> dict[str, Any]:
    key, public = keys
    return verify(receipt, opening, key, public, challenge=challenge, prover_id=prover_id, expected=expected)


def mutate_opening(opening: bytes, fn: Callable[[dict, dict], None]) -> bytes:
    index, tensors = decode_opening(opening)
    tensors = {k: v.copy() for k, v in tensors.items()}
    fn(index, tensors)
    return encode_opening(index, tensors)


def resign(receipt: dict[str, Any], engine: Engine, fn: Callable[[dict], None]) -> dict[str, Any]:
    """A malicious prover editing its own receipt and signing it again."""
    r = copy.deepcopy(receipt)
    fn(r)
    return sign_receipt(r, engine.identity.signing_key)


def recommit(engine: Engine, run: Run, fn: Callable[[Any], None], challenge: dict | None = None):
    """A malicious prover that commits to a modified trace (consistent opening)."""
    trace = copy.deepcopy(run.trace)
    fn(trace)
    com = commit_trace(trace, engine.profile)
    receipt = build_receipt(trace, com, engine.model_info, run.receipt["manifest"], engine.identity)
    ch = challenge or run.challenge
    opening = build_opening(trace, com, ch, engine.profile, engine.embed_bits, engine.embed_tree)
    return receipt, opening, ch


def dump(v: dict[str, Any]) -> str:
    return json.dumps({k: v[k] for k in ("result", "reason", "message", "layer", "position")}, default=str)
