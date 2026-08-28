# 🚀 Efficient LLM Processing and Fine-Tuning

## 🧠 Overview
This project explores how **large language models can be trained and deployed efficiently on limited hardware**—balancing **memory, speed, and accuracy**.  
I built a full experimental pipeline around **Meta’s LLaMA 3.2 1B** model to evaluate optimization methods like **KV caching, LoRA, gradient accumulation, mixed precision, and activation checkpointing**.

> 💡 The goal: make billion-parameter models *trainable and deployable* on a single 16 GB GPU without losing performance.

## 🧩 Motivation
Most LLM work assumes unlimited compute. I wanted to prove that *careful systems engineering*—not bigger clusters—unlocks scalability.  
This project merges my interests in **ML systems optimization** (Adobe, Apple AIML) and **efficient model design** from my published research on *LLM-based music recommendation* (ACM RecSys 2024).

## ⚙️ Key Contributions
| Area | Summary | Impact |
|------|----------|--------|
| **KV Cache Profiling** | Removed and re-implemented caching in LLaMA to analyze attention bottlenecks | Quantified a 6.7× runtime gap between cached and cache-free inference |
| **LoRA Implementation** | Added low-rank adapters to Q/V projection layers with r = 16, α = 32 | Reduced trainable parameters → 0.2 % of full model |
| **Mixed Precision + AMP** | Integrated FP16/FP32 hybrid training with dynamic loss scaling | ~1.8× runtime speedup, ~50 % less memory |
| **Gradient Accumulation** | Simulated large-batch updates under memory limits | Stable training on < 16 GB VRAM |
| **Checkpointing** | Applied selective activation checkpointing for transformer blocks | Freed ~40 % activation memory |

## ⚡ Inference Benchmarks

> **These numbers are not trustworthy and have not been re-measured.** They were
> taken with `time.time()` around asynchronous CUDA work with no
> `torch.cuda.synchronize()`, so they record when work was *queued*, not when it
> finished. `benchmark_inference.py` now synchronizes, but re-running requires the
> `llama/` package and the LLaMA-3.2-1B checkpoint, neither of which is in this
> repo. Treat the table below as unverified until it is re-run.


| **Batch** | **Cache** | **Peak Memory (GB)** | **Runtime (s)** | **Δ Runtime** |
|-----------:|:----------:|:--------------------:|:---------------:|:-------------:|
| 1 | ✅ ON | 3.07 | 0.37 | — |
| 1 | ❌ OFF | 3.23 | 0.41 | +11 % |
| 8 | ✅ ON | 4.50 | 0.52 | — |
| 8 | ❌ OFF | 5.76 | 1.80 | +246 % |
| 16 | ✅ ON | 6.13 | 0.63 | — |
| 16 | ❌ OFF | 8.64 | 4.25 | +574 % |

> **Insight:** KV caching is essential for scalable inference—without it, attention recomputation scales linearly with prompt length × batch size.

## Fused decode-attention CUDA kernel

A hand-written FP16 CUDA kernel for the decode step (generating token N+1 given a
KV cache of N tokens), benchmarked against PyTorch SDPA and FlashAttention-2.

**Every number below: NVIDIA L4 (24 GB, sm_89, 300 GB/s vendor peak), CUDA 12.4,
torch 2.6.0+cu124, float16, LLaMA-3.2-1B attention geometry (32 heads, 8 KV
heads, head_dim 64).**

### Why fuse

Decode attention is bandwidth bound, not compute bound: for a single query token
it streams `2·S·D` halfs of K and V and does only `O(S·D)` FLOPs on them. An
unfused implementation writes the `[B,H,S]` score matrix to HBM after QK^T, reads
it back for softmax, writes it, and reads it a third time for PV — three extra
round trips over data it could have kept in registers. The fused kernel touches
HBM once per K element and once per V element, which is the theoretical floor.
That is why the efficiency metric here is **% of peak DRAM bandwidth** rather
than FLOP/s: FLOP/s would flatter a kernel that does almost no arithmetic.

### Results

Interleaved, order rotated each round, 3 rounds x 150 iterations, 25 warmup.
Speedup is vs PyTorch `scaled_dot_product_attention` on the same shape.

| batch | ctx | impl | p50 (ms) | p99 (ms) | GB/s | % of peak | vs SDPA |
|------:|----:|------|---------:|---------:|-----:|----------:|--------:|
| 8  | 4096 | **fused**  | **0.2724** | 0.4054 | **246.4** | **82.1%** | **9.98x** |
| 8  | 4096 | flash2     | 0.3011 | 0.3932 | 222.9 | 74.3% | 9.03x |
| 8  | 4096 | sdpa       | 2.7177 | 2.7986 | 24.7  | 8.2%  | 1.00x |
| 16 | 4096 | **fused**  | 0.7086 | 0.8428 | 189.4 | 63.1% | 7.32x |
| 16 | 4096 | flash2     | 0.7291 | 0.7721 | 184.1 | 61.4% | 7.12x |
| 32 | 4096 | fused      | 1.6077 | 1.7336 | 167.0 | 55.7% | 6.16x |
| 32 | 4096 | **flash2** | **1.3005** | 1.3210 | 206.4 | 68.8% | 7.61x |
| 32 | 1024 | **fused**  | 0.3185 | 0.4413 | 210.7 | 70.2% | 8.02x |
| 32 | 1024 | flash2     | 0.3420 | 0.4157 | 196.2 | 65.4% | 7.47x |

The kernel beats FlashAttention-2 at batch 8 and 32/ctx-1024, and loses to it at
batch 32 / ctx 4096. It is 6-10x faster than PyTorch SDPA everywhere.

### Design

- One thread block per `(batch, head, split)`; 4 warps of 32 threads.
- Q is loaded once into registers — it is otherwise re-read `S` times.
- K and V rows are read as `half2`, so the 32 lanes of a warp cover 128
  contiguous bytes per step.
- Each warp keeps its own running softmax state (`max`, `sum`, and a distributed
  `acc`) in registers and walks a stride-disjoint slice of the sequence. The four
  partial states merge once at the end via the FlashAttention rescaling identity.
- Accumulation is in float; only the final store is fp16.
- Grouped-query aware: `kv_head = head / (H / H_kv)`.
- **Split-K (FlashDecoding).** With one block per `(batch, head)` the grid is
  `B·H` blocks — at batch 1 that is 32 blocks against the L4's 58 SMs, so half
  the GPU idles regardless of inner-loop quality. The sequence is split across
  blocks and a second kernel merges the per-split states.

### How the split count was chosen

Not guessed. `ncu` at batch 8 / ctx 4096 with a 256-block grid:

```
dram__throughput.avg.pct_of_peak_sustained_elapsed    46.31 %
launch__registers_per_thread                          51 register/thread
sm__warps_active.avg.pct_of_peak_sustained_active     36.37 %
```

4.4 blocks/SM x 4 warps / 48 max warps = 36.8%, matching the measured 36.37%.
So occupancy was limited by *grid size*, not by the 51 registers/thread. Sweeping
the split count at that shape:

| splits | 1 | 2 | 4 | 8 | 16 |
|--------|---|---|---|---|----|
| p50 ms | 0.3953 | **0.2847** | 0.3154 | 0.2929 | 0.3052 |
| % peak | 56.6 | **78.6** | 70.9 | 76.4 | 73.3 |

At batch 32 the grid is already 1024 blocks and every extra split cost time
(53.7% -> 51.7% -> 49.5%). So the heuristic targets a **total block count**
(8 blocks/SM), not a batch-size threshold. Batch 1 / ctx 4096 improved from
0.1843 ms to 0.0614 ms — a 3x win entirely from filling the machine.

### Correctness

- `tests/test_decode_attention.py` — **77 tests, all passing** against
  `torch.nn.functional.scaled_dot_product_attention`, max abs error gate 1e-2.
- `csrc/test_standalone.cu` — 26 cases against a float64 CPU reference,
  **max abs error 2.4e-4** (40x inside the gate). `nvcc` only, no torch.

The tests worth reading are the ones aimed at where this kind of kernel actually
breaks: sequence lengths straddling the 16-token iteration boundary, ragged
lengths within a batch, a test that writes garbage past every `seq_len` and
asserts the output is bit-identical, and score distributions large enough that a
non-online softmax returns NaN while passing everything else.

### Measurement caveats (read before quoting a number)

1. **The `% of peak` column is only a DRAM number when the data comes from DRAM.**
   The L4 has a 48 MiB L2. Shapes whose KV working set (`2048·B·S` bytes) fits in
   it are served from cache — FlashAttention-2 measured *117% of peak DRAM
   bandwidth* at batch 16 / ctx 1024 before this was caught. The benchmark now
   prints the working set and reports `n/a` for L2-resident shapes. Every row in
   the table above is >= 64 MiB and DRAM-resident.
2. **Sustained load throttles this card.** The same shape measured 0.42 ms deep
   in a long sweep and 0.28 ms in isolation; the L4 is a 72 W part and clocks
   drop as it heats (50C -> 56C across a run). Ratios within a single table stay
   fair, but headline numbers come from `bench/ab.py`, which interleaves
   implementations and rotates their order. Even so, in the batch-32/ctx-4096 row
   the fused kernel ran at a median 840 MHz against FlashAttention-2's 1245 MHz,
   so part of that particular gap is thermal rather than algorithmic.
3. p50/p99 are per-call, timed with individual CUDA event pairs — not a
   loop-level mean, which would hide the tail.

### Running it

```bash
pip install -e .                      # or let custom_attn JIT-build on import
python -m pytest tests/ -q            # correctness gate (needs a CUDA GPU)
python bench/bench_decode_attention.py --json out.json   # full sweep
python bench/ab.py                    # interleaved head-to-head
bash bench/profile_ncu.sh 8 4096      # Nsight Compute (needs sudo for counters)
```

`CUSTOM_ATTN_SPLITS=N` forces the split count, for re-running the sweep above.

## Paged KV cache

The contiguous cache preallocates `max_batch_size x max_seq_len` and holds
`max_seq_len - actual_len` of dead space per sequence, permanently. Paging hands
out fixed-size blocks as a sequence grows, so the cache costs what the sequences
actually use. This is the problem vLLM's PagedAttention solves.

**All numbers: NVIDIA L4, float16, LLaMA-3.2-1B geometry, 16 tokens/block.**

### Concurrency at fixed VRAM (8 GiB KV budget, max_seq_len 4096)

| length distribution | mean length | contiguous | paged | gain |
|---|---:|---:|---:|---:|
| short-tailed (lognormal) | 506 | 1024 | **8167** | **7.98x** |
| uniform(1, 4096) | 2072 | 1024 | 2015 | 1.97x |
| all sequences at max | 4096 | 1024 | 1024 | 1.00x |

The win is exactly the ratio of `max_seq_len` to the mean length actually used.
The last row is the case where paging buys nothing, and it is reported rather
than omitted: if every sequence really runs to `max_seq_len`, preallocation was
not wasting anything to begin with.

### What paging costs

| batch | ctx | contiguous p50 | paged p50 | overhead |
|---:|---:|---:|---:|---:|
| 8 | 1024 | 0.0645 ms | 0.0626 ms | ~0% |
| 8 | 4096 | 0.2755 ms | 0.339 ms | +23% |
| 32 | 1024 | 0.2836 ms | 0.357 ms | +26% |

Roughly 25% at long context, nothing at short context. Run-to-run spread on this
card is wide enough (see the thermal caveat above) that the 23% and 26% figures
should be read as "about a quarter", not to the percentage point.

### Getting there: the indirection cost started at +160%

The first working version was 2.6x slower than the contiguous kernel. Three
hypotheses, checked in order:

1. **Pool layout / locality.** With `[num_blocks, Hkv, BLOCK, D]`, one kv_head's
   consecutive chunks sit `Hkv*BLOCK*D*2` bytes apart -- BLOCK=16 reads 2 KiB
   then skips 14 KiB. Plausible, and wrong: sweeping the block size gave +160%,
   +157%, +158%, +155% for 16/32/64/128. Flat. Not locality.
2. **Occupancy or register pressure.** Also wrong. `ncu` on the paged kernel:
   71.84% achieved occupancy and 47 registers/thread, against 51 registers for
   the contiguous kernel. Paging used *fewer* registers and had healthy occupancy.
3. **Stall reasons.** This was it:

```
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld   4.00 sector/request
smsp__average_warps_issue_stalled_long_scoreboard                7.94 inst
smsp__average_warps_issue_stalled_short_scoreboard               1.71 inst
```

4.00 sectors/request is 128 B per warp request -- perfectly coalesced, so the
gather was not the problem. But `short_scoreboard` (shared-memory wait) was 1.71,
where the contiguous kernel has essentially none. Every global load was queued
behind its own block-table read, and the kernel did eight of those per tile (four
tokens x K and V) when a four-token tile lives entirely inside one 16-token
block. Rounding each split up to a whole block guarantees that, so the lookup
hoists to **one per tile**: +160% -> +23%.

### Block size

16 tokens is the default and the best of the four supported sizes on both axes:

| block | concurrent seqs (short-tailed) | vs contiguous |
|---:|---:|---:|
| 16 | 8166 | 7.97x |
| 32 | 8044 | 7.86x |
| 64 | 7808 | 7.62x |
| 128 | 7414 | 7.24x |

Total token capacity is identical across block sizes; the difference is internal
fragmentation in each sequence's last, partly-filled block. Latency overhead is
flat across all four, so there is nothing to trade for.

### Prefix sharing

`fork()` points a new sequence at the parent's blocks and bumps their refcounts,
so two sequences with a common prompt prefix cost one copy of it. The first write
into a block with refcount > 1 copies that block first (copy-on-write), which is
what keeps the fork from corrupting its parent. Tested directly: writing through
one sequence leaves the other's data untouched, freeing one sequence does not
release blocks the other still holds, and a forked sequence attends exactly as it
would with a private copy.

### Correctness

100 tests pass. The allocator's invariants -- free-list integrity, no block
handed out twice, rollback on a failed allocation, refcounts, copy-on-write --
run on CPU and need no GPU. The kernel parity tests assert the paged output
matches the contiguous kernel at every supported block size.

One note on those: paged output is *not* bit-identical to contiguous, and should
not be. Block-aligning each split changes where the sequence is partitioned,
which changes the order the per-split softmax states are summed. Measured
difference is 1.5e-5 to 6.1e-5, and paged and contiguous sit the same distance
(4.88e-4) from SDPA -- so the tests assert tight closeness plus "paging did not
move us further from SDPA than the contiguous kernel", which is the real claim.

```bash
python -m pytest tests/test_paged_attention.py -q   # allocator tests run on CPU
python bench/bench_paged.py --budget-gb 8
```

## 🎯 Fine-Tuning Results

- **Dataset:** Alpaca subset (200 samples)  
- **Hardware:** NVIDIA P100 (16 GB VRAM)  
- **Optimizer:** SGD (lr = 1e-5, accum = 8)  
- **LoRA Config:** r = 16, α = 32, dropout = 0.05  

| Technique | Peak Mem (MB) | Runtime/Step (s) | Notes |
|------------|---------------|------------------|-------|
| Baseline (FP32) | 14800 | 1.12 | Full-precision fine-tuning |
| + Mixed Precision | 7800 | 0.61 | 2× faster |
| + Checkpointing | 6200 | 0.73 | 40 % less VRAM |
| + LoRA (PEFT) | 6100 | 0.68 | Only 0.2 % params trainable |

Loss curves consistently decreased → confirmed correct gradient flow and numerical stability.

## 🧩 Insights
- **Compute ↔ Memory Trade-off:** Activation checkpointing and gradient accumulation are complementary; together they make billion-parameter training feasible.  
- **LoRA Generalization:** Preserves base-model knowledge while adapting quickly to new tasks.  
- **Mixed Precision Reliability:** AMP maintained stability across all configurations without underflow.  
- **Scalability:** Single-GPU runs match multi-GPU setups in efficiency per TFLOP when properly tuned.

## 🛠️ Tech Stack
**Languages:** Python, CUDA (`csrc/decode_attention.cu` — fused FP16 decode attention)  
**Frameworks:** PyTorch (AMP, Checkpointing), LoRA (PEFT)  
**Hardware:** NVIDIA P100 (16 GB)  
**Dataset:** Stanford Alpaca subset  
**Model:** Meta LLaMA 3.2 1B (decoder-only transformer)  

## 📈 Future Directions
- Extend to **quantization-aware training (QAT)** for 4-bit fine-tuning  
- Profile **attention kernel fusion** and **Flash-Attention 2** on larger LLaMA variants  
- Integrate **RLHF or DPO** for alignment-style fine-tuning  
- Build a **web demo** for side-by-side inference comparison (cached vs non-cached)
