"""Chat tokenizers: the Hugging Face tokenizer of a real checkpoint, or the tiny byte-level stand-in.

Both expose the same small interface: ``apply_chat(messages, thinking) -> token ids``, ``decode``,
the EOS ids, and the two hashes bound into the manifest.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from vgemma.canon import H

TINY_TOKENIZER_FILE = "vg_tokenizer.json"
TINY_CHAT_TEMPLATE = (
    "<bos>{% for m in messages %}<start_of_turn>{{ 'model' if m.role == 'assistant' else m.role }}\n"
    "{{ m.content }}<end_of_turn>\n{% endfor %}<start_of_turn>model\n"
)


class ChatTokenizer(Protocol):
    eos_token_ids: list[int]
    tokenizer_hash: str
    chat_template_hash: str

    def apply_chat(self, messages: list[dict[str, str]], thinking: bool = False) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...


class TinyByteTokenizer:
    """Byte-level stand-in for tests: specials 0..15, byte ``b`` is token ``16 + b``."""

    PAD, EOS, BOS, START_TURN, END_TURN = 0, 1, 2, 3, 4
    BYTE_OFFSET = 16

    def __init__(self, vocab_size: int = 512):
        if vocab_size < self.BYTE_OFFSET + 256:
            raise ValueError("tiny tokenizer needs vocab_size >= 272")
        self.vocab_size = vocab_size
        self.spec = {
            "type": "vg-byte",
            "vocab_size": vocab_size,
            "byte_offset": self.BYTE_OFFSET,
            "special": {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<start_of_turn>": 3, "<end_of_turn>": 4},
            "chat_template": TINY_CHAT_TEMPLATE,
        }
        self.eos_token_ids = [self.EOS, self.END_TURN]
        self.tokenizer_hash = H("vg/tokenizer", json.dumps(self.spec, sort_keys=True).encode()).hex()
        self.chat_template_hash = H("vg/chat_template", TINY_CHAT_TEMPLATE.encode()).hex()

    def save(self, directory: Path) -> None:
        (Path(directory) / TINY_TOKENIZER_FILE).write_text(json.dumps(self.spec, indent=1, sort_keys=True))

    @classmethod
    def load(cls, directory: Path) -> TinyByteTokenizer:
        spec = json.loads((Path(directory) / TINY_TOKENIZER_FILE).read_text())
        return cls(vocab_size=int(spec["vocab_size"]))

    def encode(self, text: str) -> list[int]:
        return [self.BYTE_OFFSET + b for b in text.encode("utf-8")]

    def apply_chat(self, messages: list[dict[str, str]], thinking: bool = False) -> list[int]:
        ids = [self.BOS]
        if thinking:
            messages = [{"role": "system", "content": "<|think|>"}, *messages]
        for m in messages:
            role = "model" if m["role"] == "assistant" else m["role"]
            ids += [self.START_TURN, *self.encode(f"{role}\n{m['content']}"), self.END_TURN, *self.encode("\n")]
        return ids + [self.START_TURN, *self.encode("model\n")]

    def decode(self, ids: list[int]) -> str:
        out = bytearray()
        for t in ids:
            if self.BYTE_OFFSET <= t < self.BYTE_OFFSET + 256:
                out.append(t - self.BYTE_OFFSET)
        return out.decode("utf-8", errors="replace")


class HFChatTokenizer:
    """The checkpoint's own tokenizer and chat template (thinking off unless requested)."""

    def __init__(self, model_dir: Path, eos_token_ids: list[int]):
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(str(model_dir))
        tok_file = Path(model_dir) / "tokenizer.json"
        raw = tok_file.read_bytes() if tok_file.exists() else json.dumps(self.tok.get_vocab(), sort_keys=True).encode()
        self.tokenizer_hash = H("vg/tokenizer", raw).hex()
        template = self.tok.chat_template or ""
        self.chat_template_hash = H("vg/chat_template", template.encode()).hex()
        self.eos_token_ids = sorted(set(eos_token_ids))

    def apply_chat(self, messages: list[dict[str, str]], thinking: bool = False) -> list[int]:
        out: Any = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, enable_thinking=thinking
        )
        if isinstance(out, dict) or hasattr(out, "input_ids"):
            out = out["input_ids"]
        return [int(t) for t in out]

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True)


def load_tokenizer(model_dir: Path, eos_token_ids: list[int]) -> ChatTokenizer:
    if (Path(model_dir) / TINY_TOKENIZER_FILE).exists():
        return TinyByteTokenizer.load(model_dir)
    return HFChatTokenizer(model_dir, eos_token_ids)
