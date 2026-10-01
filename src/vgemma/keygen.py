"""Verifier key generation from a public checkpoint (TECH_SPEC section 7).

Outputs ``key.npz`` (secret: Freivalds vectors, precomputed ``v = R W``, norm weights, layer
scalars) and ``public.json`` (roots, config hash, profile). Tensors are streamed from the
safetensors files; peak memory is one matrix in float64.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import time
from pathlib import Path
from typing import Any

import numpy as np

from vgemma.canon import bf16_to_f32, bf16_to_f64, canonical_json
from vgemma.merkle import MerkleTree
from vgemma.model import eos_token_ids, iter_checkpoint_tensors, load_profile, resolve_model_dir, text_prefix
from vgemma.profile import FAMILY_MODULE, LAYER_NORMS, GemmaProfile, UnsupportedModel
from vgemma.protocol import embedding_leaves, weight_leaf
from vgemma.tokenizer import load_tokenizer

KEY_VERSION = 1
DEFAULT_K = 16  # independent Freivalds vectors per matrix family (TECH_SPEC section 11)
LM_CHUNK = 16384


def rademacher(key_seed: bytes, label: str, k: int, m: int) -> np.ndarray:
    """``k x m`` matrix of +-1 (int8) from SHAKE-256 (a CSPRNG) keyed by the secret seed."""
    n = k * m
    stream = hashlib.shake_256(b"vg/freivalds\x00" + key_seed + label.encode()).digest((n + 7) // 8)
    bits = np.unpackbits(np.frombuffer(stream, dtype=np.uint8))[:n]
    return (bits.astype(np.int8) * 2 - 1).reshape(k, m)


def _layer_tensor_re(prefix: str) -> re.Pattern[str]:
    return re.compile("^" + re.escape(prefix) + r"layers\.(\d+)\.(.+)$")


def keygen(model: str, out: Path, k: int = DEFAULT_K, public_out: Path | None = None, log=print) -> dict[str, Any]:
    t0 = time.time()
    model_dir = resolve_model_dir(model)
    profile, cfg = load_profile(model_dir)
    eos = eos_token_ids(model_dir, cfg)
    tok = load_tokenizer(model_dir, eos)
    names = [n for n, _ in _names_only(model_dir)]
    prefix = text_prefix(names)
    _check_family_presence(profile, prefix, set(names))

    key_seed = secrets.token_bytes(32)
    layer_re = _layer_tensor_re(prefix)
    suffix_to_family = {v + ".weight": f for f, v in FAMILY_MODULE.items()}
    norm_suffixes = {n + ".weight": n for n in LAYER_NORMS}

    arrays: dict[str, np.ndarray] = {}
    weight_leaves: list[bytes] = []
    embedding_root = None
    for name, arr in iter_checkpoint_tensors(model_dir):
        weight_leaves.append(weight_leaf(name, arr))
        m = layer_re.match(name)
        if m:
            layer, suffix = int(m[1]), m[2]
            if suffix in suffix_to_family:
                fam = suffix_to_family[suffix]
                _expect_bf16(name, arr, profile.family_shape(layer, fam))
                r = rademacher(key_seed, f"{layer}/{fam}", k, arr.shape[0])
                arrays[f"r.{layer}.{fam}"] = r
                arrays[f"v.{layer}.{fam}"] = r.astype(np.float64) @ bf16_to_f64(arr)
            elif suffix in norm_suffixes:
                arrays[f"w.{layer}.{norm_suffixes[suffix]}"] = bf16_to_f32(arr)
            elif suffix == "layer_scalar":
                arrays[f"s.{layer}"] = bf16_to_f32(arr).reshape(-1)
        elif name == prefix + "norm.weight":
            arrays["w.final"] = bf16_to_f32(arr)
        elif name == prefix + "embed_tokens.weight":
            _expect_bf16(name, arr, (profile.vocab_size, profile.hidden_size))
            embedding_root = MerkleTree(embedding_leaves(arr)).root
            r_lm = rademacher(key_seed, "lm_head", k, profile.vocab_size)
            v_lm = np.zeros((k, profile.hidden_size), dtype=np.float64)
            for s in range(0, profile.vocab_size, LM_CHUNK):
                v_lm += r_lm[:, s : s + LM_CHUNK].astype(np.float64) @ bf16_to_f64(arr[s : s + LM_CHUNK])
            arrays["r.lm"], arrays["v.lm"] = r_lm, v_lm
        elif name.startswith(prefix) and name.endswith("lm_head.weight"):
            raise UnsupportedModel("checkpoint stores an untied lm_head; the gemma4 profile requires tied embeddings")

    _check_key_complete(profile, arrays)
    if embedding_root is None:
        raise UnsupportedModel("embed_tokens.weight not found")
    weights_root = MerkleTree(weight_leaves).root

    public = {
        "version": KEY_VERSION,
        "model_id": model if not Path(model).is_dir() else model_dir.name,
        "revision": model_dir.name if model_dir.parent.name == "snapshots" else "local",
        "weights_root": weights_root.hex(),
        "embedding_root": embedding_root.hex(),
        "config_hash": profile.config_hash(),
        "n_weight_tensors": len(weight_leaves),
        "text_prefix": prefix,
        "eos_token_ids": sorted(eos),
        "tokenizer_hash": tok.tokenizer_hash,
        "chat_template_hash": tok.chat_template_hash,
        "profile": profile.to_dict(),
    }
    meta = {**public, "freivalds_k": k, "key_id": hashlib.sha256(key_seed).hexdigest()[:16]}
    arrays["meta"] = np.frombuffer(canonical_json(meta).encode(), dtype=np.uint8)

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    key_path = out / "key.npz"
    np.savez(key_path, **arrays)
    public_path = Path(public_out) if public_out else out / "public.json"
    public_path.parent.mkdir(parents=True, exist_ok=True)
    elapsed = time.time() - t0
    key_bytes = key_path.stat().st_size
    public["keygen"] = {"seconds": round(elapsed, 3), "key_bytes": key_bytes, "freivalds_k": k}
    public_path.write_text(json.dumps(public, indent=1, sort_keys=True))
    n_fam = {lt: len(profile.families(profile.layer_types.index(lt))) for lt in sorted(set(profile.layer_types))}
    log(
        f"keygen: {profile.num_layers} layers, families per layer type {n_fam}, k={k}, "
        f"key {key_bytes / 2**20:.2f} MiB, {elapsed:.1f} s"
    )
    log(f"  weights_root {weights_root.hex()}  embedding_root {embedding_root.hex()}")
    log(f"  config_hash  {profile.config_hash()}")
    log(f"  key -> {key_path} (secret)   public -> {public_path}")
    return public


def _names_only(model_dir: Path):
    from safetensors import safe_open

    from vgemma.model import checkpoint_files

    for f in checkpoint_files(model_dir):
        with safe_open(str(f), framework="np") as h:
            for k in h.keys():  # noqa: SIM118
                yield k, None


def _expect_bf16(name: str, arr: np.ndarray, shape: tuple[int, ...]) -> None:
    if arr.dtype != np.uint16:
        raise UnsupportedModel(f"{name}: expected bfloat16 weights, got {arr.dtype}")
    if tuple(arr.shape) != tuple(shape):
        raise UnsupportedModel(f"{name}: shape {tuple(arr.shape)} != profile {tuple(shape)}")


def _check_family_presence(profile: GemmaProfile, prefix: str, names: set[str]) -> None:
    for layer in range(profile.num_layers):
        for fam, mod in FAMILY_MODULE.items():
            present = f"{prefix}layers.{layer}.{mod}.weight" in names
            expected = fam in profile.families(layer)
            if present != expected:
                raise UnsupportedModel(
                    f"layer {layer} ({profile.layer_types[layer]}): {mod} present={present}, expected {expected}"
                )
    if f"{prefix}layers.{profile.num_layers}.input_layernorm.weight" in names:
        raise UnsupportedModel("checkpoint has more decoder layers than num_hidden_layers")


def _check_key_complete(profile: GemmaProfile, arrays: dict[str, np.ndarray]) -> None:
    missing = []
    for layer in range(profile.num_layers):
        missing += [f"v.{layer}.{f}" for f in profile.families(layer) if f"v.{layer}.{f}" not in arrays]
        missing += [f"w.{layer}.{n}" for n in LAYER_NORMS if f"w.{layer}.{n}" not in arrays]
        if f"s.{layer}" not in arrays:
            arrays[f"s.{layer}"] = np.ones(1, dtype=np.float32)  # checkpoints without the buffer
        if profile.is_global(layer) and f"v.{layer}.wv" in arrays:
            raise UnsupportedModel(f"global layer {layer} must not have a Wv family")
    if "w.final" not in arrays:
        missing.append("w.final")
    if missing:
        raise UnsupportedModel(f"checkpoint is missing tensors needed for the key: {missing[:8]}")
