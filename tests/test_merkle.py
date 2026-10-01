from __future__ import annotations

import hashlib

import pytest

from vgemma.merkle import MerkleTree, depth_for, verify_proof


def leaves(n: int) -> list[bytes]:
    return [hashlib.sha256(f"leaf{i}".encode()).digest() for i in range(n)]


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 7, 8, 9, 16, 17, 100])
def test_proof_round_trip(n):
    ls = leaves(n)
    tree = MerkleTree(ls)
    for i in range(n):
        proof = tree.proof(i)
        assert len(proof) == depth_for(n)
        assert verify_proof(ls[i], i, n, proof, tree.root)


def test_tampered_sibling_rejected():
    ls = leaves(13)
    tree = MerkleTree(ls)
    proof = tree.proof(6)
    for k in range(len(proof)):
        bad = list(proof)
        bad[k] = bytes([bad[k][0] ^ 1]) + bad[k][1:]
        assert not verify_proof(ls[6], 6, 13, bad, tree.root)


def test_wrong_index_length_or_leaf_rejected():
    ls = leaves(10)
    tree = MerkleTree(ls)
    proof = tree.proof(3)
    assert not verify_proof(ls[3], 4, 10, proof, tree.root)
    assert not verify_proof(ls[3], 3, 10, proof[:-1], tree.root)
    assert not verify_proof(ls[3], 3, 10, [*proof, ls[0]], tree.root)
    assert not verify_proof(ls[4], 3, 10, proof, tree.root)
    assert not verify_proof(ls[3], 10, 10, proof, tree.root)  # padding slot is not a leaf


def test_padding_does_not_collide_with_real_leaves():
    assert MerkleTree(leaves(3)).root != MerkleTree(leaves(4)).root
    with pytest.raises(ValueError):
        MerkleTree([])
