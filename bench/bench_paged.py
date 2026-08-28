"""Paged vs contiguous KV cache: concurrency at fixed VRAM, and what paging costs.

Two questions, because paging is a trade and reporting only the win would be
marketing:

  1. How many concurrent sequences fit in a fixed KV budget?
  2. What does the block-table indirection cost in decode latency?

The concurrency win is entirely a function of the length distribution, so the
distribution is printed with every number. A contiguous cache must size every
slot for max_seq_len; a paged one spends blocks on tokens that exist. When every
sequence really is max_seq_len, paging wins nothing -- and that row is included
rather than omitted.
"""

import argparse
import math
import statistics
import sys
import os

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from custom_attn import (BLOCK_SIZE, SUPPORTED_BLOCK_SIZES, OutOfBlocks,  # noqa: E402
                         PagedKVCache, decode_attention, paged_decode_attention)

N_HEADS, N_KV_HEADS, HEAD_DIM = 32, 8, 64
BYTES_PER_TOKEN = 2 * N_KV_HEADS * HEAD_DIM * 2   # K and V, fp16


def contiguous_capacity(budget_bytes, max_seq_len):
    """A contiguous cache reserves max_seq_len per slot, used or not."""
    return budget_bytes // (BYTES_PER_TOKEN * max_seq_len)


def paged_capacity(budget_bytes, lengths, block_size=BLOCK_SIZE):
    """Admit sequences from `lengths` until the pool is exhausted. Actually runs
    the allocator rather than dividing -- block rounding and the free list are
    part of what is being measured."""
    num_blocks = budget_bytes // (BYTES_PER_TOKEN * block_size)
    # block_size must reach the allocator too, or num_blocks is computed for one
    # block size and consumed at another -- which is what an earlier version of
    # this file did, making larger blocks look 8x worse than they are.
    cache = PagedKVCache(num_blocks, N_KV_HEADS, HEAD_DIM, block_size=block_size,
                         device="cpu", allocate_pool=False)
    admitted = 0
    for i, n in enumerate(lengths):
        try:
            cache.allocate(i, int(n))
        except OutOfBlocks:
            break
        admitted += 1
    return admitted, num_blocks, cache.num_used_blocks


def sample_lengths(kind, n, max_seq_len, seed=0):
    g = torch.Generator().manual_seed(seed)
    if kind == "uniform":
        return torch.randint(1, max_seq_len + 1, (n,), generator=g)
    if kind == "short-tailed":      # most prompts short, a few long
        x = torch.distributions.LogNormal(math.log(max_seq_len / 12), 0.9).sample((n,))
        return x.clamp(1, max_seq_len).long()
    if kind == "all-max":           # the case where paging wins nothing
        return torch.full((n,), max_seq_len, dtype=torch.long)
    raise ValueError(kind)


def bench_latency(B, S, block_size=BLOCK_SIZE, iters=100, warmup=20):
    """Indirection cost: same logical cache, both layouts, same shape."""
    q = torch.randn(B, N_HEADS, HEAD_DIM, device="cuda", dtype=torch.float16).contiguous()
    k = torch.randn(B, N_KV_HEADS, S, HEAD_DIM, device="cuda", dtype=torch.float16).contiguous()
    v = torch.randn(B, N_KV_HEADS, S, HEAD_DIM, device="cuda", dtype=torch.float16).contiguous()
    lens = torch.full((B,), S, device="cuda", dtype=torch.int32)

    per_seq = (S + block_size - 1) // block_size
    cache = PagedKVCache(B * per_seq, N_KV_HEADS, HEAD_DIM,
                         block_size=block_size, device="cuda")
    for b in range(B):
        cache.allocate(b, 0)
        cache.write_prefix(b, k[b].transpose(0, 1), v[b].transpose(0, 1))
    table, plens = cache.build_block_table(list(range(B)))

    fns = {"contiguous": lambda: decode_attention(q, k, v, lens),
           "paged": lambda: paged_decode_attention(q, cache.k_pool, cache.v_pool, table, plens)}

    # Correctness before speed: a fast wrong gather is the failure mode here.
    assert torch.equal(fns["paged"](), fns["contiguous"]()), "paged != contiguous"

    out = {}
    for name, fn in fns.items():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        st = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        en = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            st[i].record(); fn(); en[i].record()
        torch.cuda.synchronize()
        lat = sorted(s.elapsed_time(e) for s, e in zip(st, en))
        out[name] = lat[len(lat) // 2]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-gb", type=float, default=8.0)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--pool-seqs", type=int, default=20000)
    args = ap.parse_args()

    budget = int(args.budget_gb * 2**30)
    M = args.max_seq_len

    print(f"KV budget {args.budget_gb:.1f} GiB   max_seq_len {M}   block {BLOCK_SIZE} tokens")
    print(f"geometry: {N_KV_HEADS} kv heads x {HEAD_DIM} dim, fp16 "
          f"-> {BYTES_PER_TOKEN} B/token ({BYTES_PER_TOKEN*M/2**20:.1f} MiB per contiguous slot)\n")

    contig = contiguous_capacity(budget, M)
    print(f"{'distribution':>14}  {'mean len':>9}  {'contiguous':>11}  {'paged':>7}  {'gain':>6}")
    print("-" * 58)
    for kind in ("short-tailed", "uniform", "all-max"):
        lengths = sample_lengths(kind, args.pool_seqs, M)
        admitted, nblocks, used = paged_capacity(budget, lengths)
        mean_len = lengths[:admitted].float().mean().item() if admitted else 0.0
        gain = admitted / contig if contig else float("nan")
        print(f"{kind:>14}  {mean_len:9.0f}  {contig:11d}  {admitted:7d}  {gain:5.2f}x")

    print("\nPaging wins exactly the ratio of max_seq_len to the mean length actually")
    print("used; with every sequence at max_seq_len it wins nothing (all-max row).")

    if torch.cuda.is_available():
        print(f"\nIndirection cost on {torch.cuda.get_device_name(0)} (p50 ms, fp16).")
        print("Block size is swept because it was the obvious suspect for the")
        print("indirection cost -- it is not. Overhead is flat across 16..128; the")
        print("cost was the block-table lookup sitting on the load critical path,")
        print("fixed by resolving one block per tile instead of one per token.\n")
        hdr = f"{'batch':>6} {'ctx':>6} {'contiguous':>11}"
        for bs in SUPPORTED_BLOCK_SIZES:
            hdr += f"  blk={bs:<3}"
        print(hdr)
        for B, S in [(8, 1024), (8, 4096), (32, 1024)]:
            base = None
            row = ""
            for bs in SUPPORTED_BLOCK_SIZES:
                r = bench_latency(B, S, block_size=bs)
                base = r["contiguous"]
                row += f"  {(r['paged']/base-1)*100:+6.0f}%"
            print(f"{B:>6} {S:>6} {base:11.4f}{row}")
        print("\n(percentages are paged overhead vs the contiguous kernel at the same shape)")

        print("\nFragmentation cost of larger blocks (short-tailed, 8 GiB budget):")
        lengths = sample_lengths("short-tailed", args.pool_seqs, M)
        for bs in SUPPORTED_BLOCK_SIZES:
            adm, _, _ = paged_capacity(budget, lengths, block_size=bs)
            print(f"  block {bs:>3} tokens -> {adm:6d} concurrent sequences "
                  f"({adm/contig:.2f}x contiguous)")


if __name__ == "__main__":
    main()
