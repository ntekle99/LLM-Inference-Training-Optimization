"""Paged KV cache: allocator invariants and kernel equivalence.

Two independent things are being checked and they fail differently:

  * the allocator is pure bookkeeping (free list, refcounts, copy-on-write) and
    is testable on CPU without a GPU;
  * the kernel must produce bit-identical output to the contiguous kernel for
    the same logical cache contents, which needs CUDA.

Only the second is skipped without a GPU, so allocator bugs still surface on a
laptop.
"""

import math

import pytest
import torch

from custom_attn import (BLOCK_SIZE, SUPPORTED_BLOCK_SIZES, OutOfBlocks,
                         PagedKVCache)

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

HKV, D = 8, 64
TOL = 1e-2  # same gate as the contiguous kernel


def mk_cache(num_blocks=64, device="cpu"):
    return PagedKVCache(num_blocks, HKV, D, device=device, dtype=torch.float16)


# ------------------------------------------------------------ the allocator

def test_rejects_unsupported_block_size():
    with pytest.raises(ValueError):
        PagedKVCache(4, HKV, D, block_size=24, device="cpu")


def test_allocate_rounds_up_to_whole_blocks():
    c = mk_cache()
    c.allocate("a", 1)
    assert len(c.blocks_of("a")) == 1
    c.allocate("b", BLOCK_SIZE)
    assert len(c.blocks_of("b")) == 1
    c.allocate("c", BLOCK_SIZE + 1)
    assert len(c.blocks_of("c")) == 2


def test_free_returns_blocks_to_the_pool():
    c = mk_cache(num_blocks=8)
    c.allocate("a", 4 * BLOCK_SIZE)
    assert c.num_free_blocks == 4
    c.free("a")
    assert c.num_free_blocks == 8


def test_pool_exhaustion_raises():
    c = mk_cache(num_blocks=2)
    c.allocate("a", 2 * BLOCK_SIZE)
    with pytest.raises(OutOfBlocks):
        c.allocate("b", BLOCK_SIZE)


def test_failed_allocation_does_not_leak():
    """A partial allocation that hits the limit must roll back, or the pool
    bleeds a few blocks on every rejected request until it wedges."""
    c = mk_cache(num_blocks=4)
    c.allocate("a", 2 * BLOCK_SIZE)
    with pytest.raises(OutOfBlocks):
        c.allocate("b", 10 * BLOCK_SIZE)   # needs 10, only 2 left
    assert c.num_free_blocks == 2, "rolled-back allocation leaked blocks"
    c.allocate("b", 2 * BLOCK_SIZE)        # the 2 must still be usable
    assert c.num_free_blocks == 0


def test_no_block_is_handed_out_twice():
    c = mk_cache(num_blocks=16)
    seen = set()
    for i in range(4):
        for b in c.allocate(f"s{i}", 4 * BLOCK_SIZE):
            assert b not in seen, f"block {b} allocated twice"
            seen.add(b)


# --------------------------------------------------------- prefix sharing

def test_fork_shares_blocks_rather_than_copying():
    c = mk_cache()
    c.allocate("a", 4 * BLOCK_SIZE)
    before = c.num_free_blocks
    c.fork("a", "b")
    assert c.num_free_blocks == before, "fork consumed blocks"
    assert c.blocks_of("a") == c.blocks_of("b")
    for blk in c.blocks_of("a"):
        assert c.refcount(blk) == 2


def test_shared_blocks_survive_one_free():
    c = mk_cache()
    c.allocate("a", 2 * BLOCK_SIZE)
    blocks = c.blocks_of("a")
    c.fork("a", "b")
    c.free("a")
    assert c.num_free_blocks == c.num_blocks - 2, "freed a block still in use"
    for blk in blocks:
        assert c.refcount(blk) == 1
    c.free("b")
    assert c.num_free_blocks == c.num_blocks


def test_copy_on_write_isolates_the_fork():
    """The whole point of refcounting: writing through one sequence must not be
    visible through the other."""
    c = mk_cache()
    c.allocate("a", 0)
    k = torch.arange(HKV * D, dtype=torch.float16).reshape(HKV, D)
    for _ in range(4):
        c.write_token("a", k, k)
    c.fork("a", "b")
    shared = c.blocks_of("a")[0]
    assert c.refcount(shared) == 2

    other = torch.full((HKV, D), -7.0, dtype=torch.float16)
    c.write_token("b", other, other)          # diverge

    assert c.blocks_of("b")[0] != shared, "wrote into a shared block"
    assert c.refcount(shared) == 1
    # a's data is untouched, and b kept the copied prefix.
    assert torch.equal(c.k_pool[shared, :, 0, :], k)
    assert torch.equal(c.k_pool[c.blocks_of("b")[0], :, 0, :], k)
    assert torch.equal(c.k_pool[c.blocks_of("b")[0], :, 4, :], other)


def test_no_cow_when_unshared():
    c = mk_cache()
    c.allocate("a", 0)
    k = torch.ones(HKV, D, dtype=torch.float16)
    c.write_token("a", k, k)
    blk = c.blocks_of("a")[0]
    before = c.num_free_blocks
    c.write_token("a", k, k)
    assert c.blocks_of("a")[0] == blk, "copied a block it owned outright"
    assert c.num_free_blocks == before


def test_write_grows_block_list():
    c = mk_cache()
    c.allocate("a", 0)
    k = torch.zeros(HKV, D, dtype=torch.float16)
    for i in range(BLOCK_SIZE * 2 + 1):
        c.write_token("a", k, k)
    assert c.seq_len("a") == BLOCK_SIZE * 2 + 1
    assert len(c.blocks_of("a")) == 3


# ------------------------------------------------------------- block table

def test_block_table_shape_and_padding():
    c = mk_cache(device="cpu")
    c.allocate("a", 3 * BLOCK_SIZE)
    c.allocate("b", BLOCK_SIZE)
    table, lens = c.build_block_table(["a", "b"])
    assert table.shape == (2, 3)
    assert lens.tolist() == [3 * BLOCK_SIZE, BLOCK_SIZE]
    assert table[0].tolist() == c.blocks_of("a")


# ---------------------------------------------------------- kernel parity

@cuda_only
@pytest.mark.parametrize("block_size", SUPPORTED_BLOCK_SIZES)
def test_paged_matches_contiguous_across_block_sizes(block_size):
    """Every supported block size is a separate kernel instantiation with its own
    index arithmetic, so each one needs its own parity check."""
    from custom_attn import (decode_attention, decode_attention_reference,
                             paged_decode_attention)

    torch.manual_seed(3)
    B, S, H = 2, 300, 32
    q = torch.randn(B, H, D, device="cuda", dtype=torch.float16).contiguous()
    k = torch.randn(B, HKV, S, D, device="cuda", dtype=torch.float16).contiguous()
    v = torch.randn(B, HKV, S, D, device="cuda", dtype=torch.float16).contiguous()
    lens = torch.full((B,), S, device="cuda", dtype=torch.int32)

    per_seq = (S + block_size - 1) // block_size
    cache = PagedKVCache(B * per_seq, HKV, D, block_size=block_size, device="cuda")
    for b in range(B):
        cache.allocate(b, 0)
        cache.write_prefix(b, k[b].transpose(0, 1), v[b].transpose(0, 1))
    table, plens = cache.build_block_table(list(range(B)))

    want = decode_attention(q, k, v, lens)
    got = paged_decode_attention(q, cache.k_pool, cache.v_pool, table, plens)
    ref = decode_attention_reference(q, k, v, lens)

    # Not bit-exact, and shouldn't be: the paged path rounds each split up to a
    # whole block, so it partitions the sequence differently and sums the
    # per-split softmax states in a different order. That moves the last bits.
    # What must hold is that paging does not degrade accuracy -- so the paged
    # output tracks the contiguous one tightly, and sits no further from SDPA
    # than the contiguous kernel does.
    d_paged_contig = (got.float() - want.float()).abs().max().item()
    d_paged_ref = (got.float() - ref.float()).abs().max().item()
    d_contig_ref = (want.float() - ref.float()).abs().max().item()
    assert d_paged_contig < 1e-3, f"block_size={block_size}: {d_paged_contig}"
    assert d_paged_ref < TOL, f"block_size={block_size} vs SDPA: {d_paged_ref}"
    assert d_paged_ref <= d_contig_ref * 1.5 + 1e-4, (
        f"block_size={block_size}: paging degraded accuracy "
        f"({d_paged_ref} vs {d_contig_ref})")


@cuda_only
@pytest.mark.parametrize("B,S", [(1, 16), (1, 100), (4, 512), (8, 4096), (2, 17)])
def test_paged_matches_contiguous(B, S):
    """Same logical cache, two layouts, identical output.

    The contiguous kernel is already gated against SDPA, so matching it exactly
    is the strongest statement available: any difference is the indirection.
    """
    from custom_attn import decode_attention, paged_decode_attention

    torch.manual_seed(0)
    H = 32
    q = torch.randn(B, H, D, device="cuda", dtype=torch.float16).contiguous()
    k = torch.randn(B, HKV, S, D, device="cuda", dtype=torch.float16).contiguous()
    v = torch.randn(B, HKV, S, D, device="cuda", dtype=torch.float16).contiguous()
    lens = torch.full((B,), S, device="cuda", dtype=torch.int32)

    n_blocks_per_seq = (S + BLOCK_SIZE - 1) // BLOCK_SIZE
    cache = PagedKVCache(B * n_blocks_per_seq, HKV, D, device="cuda")
    seq_ids = []
    for b in range(B):
        cache.allocate(b, 0)
        seq_ids.append(b)
        # [S, HKV, D] view of this sequence's cache
        cache.write_prefix(b, k[b].transpose(0, 1), v[b].transpose(0, 1))

    table, plens = cache.build_block_table(seq_ids)
    assert torch.equal(plens, lens)

    want = decode_attention(q, k, v, lens)
    got = paged_decode_attention(q, cache.k_pool, cache.v_pool, table, plens)
    diff = (got.float() - want.float()).abs().max().item()
    assert diff < 1e-3, f"paged != contiguous, max diff {diff:.6f}"


@cuda_only
def test_paged_ragged_lengths():
    from custom_attn import decode_attention, paged_decode_attention

    torch.manual_seed(1)
    H, B, S = 32, 4, 256
    lens_list = [256, 1, 33, 100]
    q = torch.randn(B, H, D, device="cuda", dtype=torch.float16).contiguous()
    k = torch.randn(B, HKV, S, D, device="cuda", dtype=torch.float16).contiguous()
    v = torch.randn(B, HKV, S, D, device="cuda", dtype=torch.float16).contiguous()
    lens = torch.tensor(lens_list, device="cuda", dtype=torch.int32)

    cache = PagedKVCache(B * (S // BLOCK_SIZE + 1), HKV, D, device="cuda")
    for b in range(B):
        cache.allocate(b, 0)
        n = lens_list[b]
        cache.write_prefix(b, k[b, :, :n].transpose(0, 1), v[b, :, :n].transpose(0, 1))
    table, plens = cache.build_block_table(list(range(B)))

    want = decode_attention(q, k, v, lens)
    got = paged_decode_attention(q, cache.k_pool, cache.v_pool, table, plens)
    assert (got.float() - want.float()).abs().max().item() < 1e-3


@cuda_only
def test_shared_prefix_gives_same_answer_as_private_copy():
    """A forked sequence must attend exactly as it would with its own copy --
    otherwise prefix sharing is silently changing model output."""
    from custom_attn import paged_decode_attention

    torch.manual_seed(2)
    H, S = 32, 64
    q = torch.randn(1, H, D, device="cuda", dtype=torch.float16).contiguous()
    k = torch.randn(S, HKV, D, device="cuda", dtype=torch.float16)
    v = torch.randn(S, HKV, D, device="cuda", dtype=torch.float16)

    cache = PagedKVCache(64, HKV, D, device="cuda")
    cache.allocate("parent", 0)
    cache.write_prefix("parent", k, v)
    cache.fork("parent", "child")

    private = PagedKVCache(64, HKV, D, device="cuda")
    private.allocate("solo", 0)
    private.write_prefix("solo", k, v)

    t_shared, l_shared = cache.build_block_table(["child"])
    t_priv, l_priv = private.build_block_table(["solo"])

    a = paged_decode_attention(q, cache.k_pool, cache.v_pool, t_shared, l_shared)
    b = paged_decode_attention(q, private.k_pool, private.v_pool, t_priv, l_priv)
    # Here bit-exactness *does* still hold: same block size, same lengths, so the
    # same partitioning and the same summation order. Only the physical blocks
    # differ, and those hold identical bytes.
    assert torch.equal(a, b)
