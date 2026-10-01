"""Binary Merkle tree over 32-byte leaves with power-of-two padding.

Leaves are already domain-separated hashes (``vg/pos``, ``vg/tensor``, ``vg/weight``); interior
nodes use ``vg/node`` and padding uses a fixed empty-leaf hash, so a leaf can never be confused
with a node.
"""

from __future__ import annotations

from collections.abc import Sequence

from vgemma.canon import H

EMPTY_LEAF = H("vg/leaf", b"empty")


def node(left: bytes, right: bytes) -> bytes:
    return H("vg/node", left, right)


def depth_for(n_leaves: int) -> int:
    if n_leaves < 1:
        raise ValueError("a Merkle tree needs at least one leaf")
    return (n_leaves - 1).bit_length()


class MerkleTree:
    def __init__(self, leaves: Sequence[bytes]):
        if not leaves:
            raise ValueError("a Merkle tree needs at least one leaf")
        self.n_leaves = len(leaves)
        size = 1 << depth_for(self.n_leaves)
        level = list(leaves) + [EMPTY_LEAF] * (size - self.n_leaves)
        self.levels: list[list[bytes]] = [level]
        while len(level) > 1:
            level = [node(level[i], level[i + 1]) for i in range(0, len(level), 2)]
            self.levels.append(level)

    @property
    def root(self) -> bytes:
        return self.levels[-1][0]

    def proof(self, index: int) -> list[bytes]:
        if not 0 <= index < self.n_leaves:
            raise IndexError(f"leaf index {index} out of range [0, {self.n_leaves})")
        path = []
        for level in self.levels[:-1]:
            path.append(level[index ^ 1])
            index >>= 1
        return path


def root_of(leaves: Sequence[bytes]) -> bytes:
    return MerkleTree(leaves).root


def verify_proof(leaf: bytes, index: int, n_leaves: int, proof: Sequence[bytes], root: bytes) -> bool:
    if not 0 <= index < n_leaves or len(proof) != depth_for(n_leaves):
        return False
    acc = leaf
    for sibling in proof:
        if len(sibling) != 32:
            return False
        acc = node(sibling, acc) if index & 1 else node(acc, sibling)
        index >>= 1
    return acc == root
