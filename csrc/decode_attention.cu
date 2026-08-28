// Fused single-query (decode-step) attention.
//
// Shapes, for one call:
//   q       [B, H,   D]        the single query token being generated
//   k_cache [B, Hkv, S_max, D] contiguous KV cache
//   v_cache [B, Hkv, S_max, D]
//   lens    [B]                valid cached tokens per sequence (<= S_max)
//   out     [B, H,   D]
//
// Why fuse. Decode attention is bandwidth bound, not compute bound: for one
// query token we stream 2*S*D halfs of K and V and do only O(S*D) FLOPs on
// them. An unfused implementation writes the [B,H,S] score matrix to HBM after
// QK^T, reads it back for softmax, writes it again, and reads it a third time
// for PV. That is three extra HBM round trips over the scores to save nothing.
// Fusing keeps scores in registers, so the kernel touches HBM exactly once per
// K element and once per V element -- which is the theoretical floor, and why
// "% of peak DRAM bandwidth" is the honest metric for this kernel rather than
// FLOP/s.
//
// Parallel decomposition: one thread block per (batch, head, split). Within a
// block, warp w walks a disjoint stride-WARPS slice of that split's token range
// keeping its own running softmax state, then the WARPS partial states are
// merged with the standard FlashAttention rescaling identity.
//
// Why split at all. With one block per (batch, head) the grid is B*H blocks --
// at batch 1 on LLaMA-3.2-1B that is 32 blocks against 58 SMs, so nearly half
// the GPU idles no matter how good the inner loop is. Measured on an L4, that
// shape ran 3x slower than FlashAttention-2 while the batch-16 shape was
// competitive, which is the signature of an occupancy problem rather than a
// bandwidth one. Splitting the sequence across blocks (FlashDecoding) buys back
// the parallelism; a second kernel then merges the per-split softmax states.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kWarpSize = 32;

// Tokens each warp has in flight per iteration. The kernel is latency bound on
// the K/V loads, so we issue UNROLL independent loads before doing any math on
// them; this is what keeps enough memory requests outstanding to saturate DRAM.
constexpr int kUnroll = 4;

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int off = kWarpSize / 2; off > 0; off >>= 1) {
    v += __shfl_down_sync(0xffffffffu, v, off);
  }
  return v;
}

// HEAD_DIM halfs per token row, read as half2 by a full warp.
// VEC = half2 elements owned by each lane. Lane `l` owns dims
// [2*(l + k*32), 2*(l + k*32)+1] for k in [0, VEC).
template <int HEAD_DIM>
struct Vec {
  static constexpr int kValue = HEAD_DIM / 2 / kWarpSize;
  static_assert(HEAD_DIM % (2 * kWarpSize) == 0,
                "HEAD_DIM must be a multiple of 64");
};

// Address of token `tok`'s row for this (batch, kv_head).
//
// Contiguous: the cache is [B, Hkv, S_max, D] and the row is a stride away.
// Paged: the cache is a pool of fixed-size blocks, [num_blocks, Hkv, BLOCK, D],
// and the sequence's blocks are listed in a per-sequence block table. The token
// index splits into (block slot, offset in block) and one int32 load through the
// table gives the physical block. Within a block the BLOCK tokens of one kv_head
// stay contiguous, so a tile of consecutive tokens is still a coalesced read;
// only the indirection is new.
//
// Both paths run the same softmax body below -- the alternative was a second
// copy of the kernel, which is how the two get to disagree numerically later.
// Max block-table entries cached in shared memory per thread block. 512 entries
// = 2 KiB = 8192 tokens of context, which covers every shape here; beyond it the
// lookup falls back to global memory.
constexpr int kMaxSharedBlocks = 512;

template <int HEAD_DIM, int BLOCK, bool kPaged>
__device__ __forceinline__ const __half* row_ptr(const __half* contig,
                                                 const __half* pool,
                                                 const int* __restrict__ btable,
                                                 const int* s_btable,
                                                 int Hkv, int kv_head, int tok) {
  if constexpr (kPaged) {
    const int slot = tok / BLOCK;
    // Reading the block table from global memory here costs more than it looks:
    // the K/V address *depends* on the loaded value, so every tile stalls on a
    // dependent global load before its prefetch can even be issued. Measured at
    // +160% p50 over the contiguous kernel. Staging the table in shared memory
    // once per block turns that into a shared-memory read.
    const int phys = (slot < kMaxSharedBlocks) ? s_btable[slot] : btable[slot];
    return pool + ((static_cast<long long>(phys) * Hkv + kv_head) * BLOCK +
                   (tok % BLOCK)) * HEAD_DIM;
  } else {
    return contig + static_cast<long long>(tok) * HEAD_DIM;
  }
}

// Load one token row into per-lane registers as float2, coalesced: the 32 lanes
// of a warp cover 32 consecutive half2 = 128 contiguous bytes per step.
template <int HEAD_DIM>
__device__ __forceinline__ void load_row(const __half* __restrict__ row,
                                         int lane, float2 (&dst)[Vec<HEAD_DIM>::kValue]) {
  constexpr int kVec = Vec<HEAD_DIM>::kValue;
  const __half2* row2 = reinterpret_cast<const __half2*>(row);
#pragma unroll
  for (int k = 0; k < kVec; ++k) {
    dst[k] = __half22float2(row2[lane + k * kWarpSize]);
  }
}

template <int HEAD_DIM, int WARPS, bool kPaged = false, int BLOCK = 16>
__global__ __launch_bounds__(WARPS* kWarpSize) void decode_attention_kernel(
    const __half* __restrict__ q,
    const __half* __restrict__ k_cache,
    const __half* __restrict__ v_cache,
    const int* __restrict__ lens,
    __half* __restrict__ out,
    float* __restrict__ p_m, float* __restrict__ p_l, float* __restrict__ p_acc,
    const int* __restrict__ block_table, int max_blocks,
    int H, int Hkv, int S_max, float scale, int nsplits) {
  constexpr int kVec = Vec<HEAD_DIM>::kValue;
  constexpr int kThreads = WARPS * kWarpSize;

  const int head = blockIdx.x;
  const int batch = blockIdx.y;
  const int split = blockIdx.z;
  const int tid = threadIdx.x;
  const int warp = tid / kWarpSize;
  const int lane = tid % kWarpSize;

  // Grouped-query attention: several query heads share one KV head.
  const int kv_head = head / (H / Hkv);
  const int seq_len = lens[batch];

  __half* out_row = out + (static_cast<long long>(batch) * H + head) * HEAD_DIM;

  // This block's slice of the sequence. On the paged path the slice is rounded
  // up to a whole number of blocks, which guarantees a kUnroll-token tile lies
  // entirely inside one block -- that is what lets the block-table lookup be
  // hoisted out of the inner loop below.
  int per_split = (seq_len + nsplits - 1) / nsplits;
  if constexpr (kPaged) per_split = ((per_split + BLOCK - 1) / BLOCK) * BLOCK;
  const int tok_begin = split * per_split;
  const int tok_end = min(seq_len, tok_begin + per_split);

  const long long pidx = (static_cast<long long>(batch) * H + head) * nsplits + split;
  float* pacc_row = p_acc + pidx * HEAD_DIM;

  if (seq_len <= 0 || tok_begin >= tok_end) {
    // Empty slice. With one split this is the empty-cache case and the kernel
    // defines the output as zero; with several, the merge must see a state that
    // contributes nothing, which is l = 0 (its rescale factor becomes 0).
    if (nsplits == 1) {
      for (int i = tid; i < HEAD_DIM; i += kThreads) out_row[i] = __float2half(0.f);
    } else {
      if (tid == 0) { p_m[pidx] = -INFINITY; p_l[pidx] = 0.f; }
      for (int i = tid; i < HEAD_DIM; i += kThreads) pacc_row[i] = 0.f;
    }
    return;
  }

  // Q stays in registers for the whole kernel -- it is read S times otherwise.
  float2 qv[kVec];
  load_row<HEAD_DIM>(q + (static_cast<long long>(batch) * H + head) * HEAD_DIM, lane, qv);

  // Contiguous path: stride into this sequence's slab. Paged path: this
  // sequence's slice of the block table.
  const long long kv_base =
      (static_cast<long long>(batch) * Hkv + kv_head) * static_cast<long long>(S_max) * HEAD_DIM;
  const __half* k_base = kPaged ? nullptr : k_cache + kv_base;
  const __half* v_base = kPaged ? nullptr : v_cache + kv_base;
  const int* btable = kPaged ? block_table + static_cast<long long>(batch) * max_blocks : nullptr;

  // Stage this sequence's block table in shared memory. Sized 1 on the
  // contiguous path so it costs no occupancy there. The early-out above is
  // block-uniform, so every thread that reaches this __syncthreads() reaches it
  // together.
  constexpr int kBTSlots = kPaged ? kMaxSharedBlocks : 1;
  __shared__ int s_btable[kBTSlots];
  if constexpr (kPaged) {
    const int need = min(max_blocks, kMaxSharedBlocks);
    for (int i = tid; i < need; i += kThreads) s_btable[i] = btable[i];
    __syncthreads();
  }

  // Per-warp running softmax state. acc is distributed across the warp in the
  // same lane->dim mapping as the row loads, so no shared memory is needed
  // until the final cross-warp merge.
  float run_max = -INFINITY;
  float run_sum = 0.f;
  float2 acc[kVec];
#pragma unroll
  for (int k = 0; k < kVec; ++k) acc[k] = make_float2(0.f, 0.f);

  static_assert(BLOCK >= WARPS * kUnroll, "a tile must fit inside one block");

  for (int base = tok_begin + warp * kUnroll; base < tok_end; base += WARPS * kUnroll) {
    float2 kv_reg[kUnroll][kVec];
    int valid[kUnroll];

    // Resolve the block once per tile. Profiling the per-token version showed
    // short_scoreboard (shared-memory wait) at 1.71 stalls per issue-active:
    // every global load was queued behind its own block-table read. A tile is
    // kUnroll consecutive tokens inside one block, so one lookup serves all of
    // them, for K and V alike.
    const __half* k_tile = k_base;
    const __half* v_tile = v_base;
    int tile_off = base;
    if constexpr (kPaged) {
      const int slot = base / BLOCK;
      const int phys = (slot < kMaxSharedBlocks) ? s_btable[slot] : btable[slot];
      const long long blk =
          (static_cast<long long>(phys) * Hkv + kv_head) * BLOCK * HEAD_DIM;
      k_tile = k_cache + blk;
      v_tile = v_cache + blk;
      tile_off = base % BLOCK;
    }

    // Issue all UNROLL K loads before touching any of them.
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      const int tok = base + u;
      valid[u] = (tok < tok_end);
      if (valid[u]) {
        load_row<HEAD_DIM>(k_tile + static_cast<long long>(tile_off + u) * HEAD_DIM,
                           lane, kv_reg[u]);
      }
    }

    // V for the same tokens, issued before any math. V traffic equals K traffic
    // in this kernel, so leaving these loads inside the softmax update below
    // halved the memory parallelism -- they were serialised one token at a time
    // behind a dependency chain they do not actually have.
    float2 vv_reg[kUnroll][kVec];
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      if (valid[u]) {
        load_row<HEAD_DIM>(v_tile + static_cast<long long>(tile_off + u) * HEAD_DIM,
                           lane, vv_reg[u]);
      }
    }

    float score[kUnroll];
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      float dot = 0.f;
      if (valid[u]) {
#pragma unroll
        for (int k = 0; k < kVec; ++k) {
          dot += qv[k].x * kv_reg[u][k].x + qv[k].y * kv_reg[u][k].y;
        }
      }
      // Every lane holds a partial dot over its own dims; reduce to lane 0 and
      // broadcast, so the whole warp agrees on the score.
      dot = warp_reduce_sum(dot);
      // -inf for out-of-range tokens makes their softmax weight exactly 0.
      score[u] = valid[u] ? __shfl_sync(0xffffffffu, dot, 0) * scale : -INFINITY;
    }

    // Online softmax, rescaled once per tile rather than once per token. The
    // running max can only move at a tile boundary, so a single correction
    // covers all kUnroll tokens -- kUnroll times fewer exp() and fewer passes
    // over the accumulator, for identical arithmetic.
    float tile_max = score[0];
#pragma unroll
    for (int u = 1; u < kUnroll; ++u) tile_max = fmaxf(tile_max, score[u]);

    const float new_max = fmaxf(run_max, tile_max);
    const float correction = __expf(run_max - new_max);  // 0 on the first tile

    float p[kUnroll];
    float psum = 0.f;
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      p[u] = __expf(score[u] - new_max);
      psum += p[u];
    }

    run_sum = run_sum * correction + psum;
    run_max = new_max;

#pragma unroll
    for (int k = 0; k < kVec; ++k) {
      float x = acc[k].x * correction;
      float y = acc[k].y * correction;
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        if (valid[u]) {
          x += p[u] * vv_reg[u][k].x;
          y += p[u] * vv_reg[u][k].y;
        }
      }
      acc[k].x = x;
      acc[k].y = y;
    }
  }

  // Merge the WARPS independent softmax states.
  __shared__ float s_max[WARPS];
  __shared__ float s_sum[WARPS];
  __shared__ float s_acc[WARPS][HEAD_DIM];

  if (lane == 0) {
    s_max[warp] = run_max;
    s_sum[warp] = run_sum;
  }
#pragma unroll
  for (int k = 0; k < kVec; ++k) {
    s_acc[warp][2 * (lane + k * kWarpSize)] = acc[k].x;
    s_acc[warp][2 * (lane + k * kWarpSize) + 1] = acc[k].y;
  }
  __syncthreads();

  if (warp == 0) {
    float g_max = -INFINITY;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) g_max = fmaxf(g_max, s_max[w]);

    float g_sum = 0.f;
    float w_scale[WARPS];
#pragma unroll
    for (int w = 0; w < WARPS; ++w) {
      // A warp that saw no tokens has run_max = -inf and run_sum = 0; its
      // scale is exp(-inf) = 0, so it contributes nothing rather than NaN.
      w_scale[w] = (s_sum[w] > 0.f) ? __expf(s_max[w] - g_max) : 0.f;
      g_sum += s_sum[w] * w_scale[w];
    }

    // With one split this block owns the whole row, so normalise and emit fp16.
    // With several, hand the *unnormalised* accumulator plus (m, l) to the merge
    // kernel -- normalising here would discard the information it needs.
    const float inv = (nsplits == 1) ? 1.f / g_sum : 1.f;
    if (nsplits > 1 && lane == 0) { p_m[pidx] = g_max; p_l[pidx] = g_sum; }
#pragma unroll
    for (int k = 0; k < kVec; ++k) {
      const int d0 = 2 * (lane + k * kWarpSize);
      float x = 0.f, y = 0.f;
#pragma unroll
      for (int w = 0; w < WARPS; ++w) {
        x += s_acc[w][d0] * w_scale[w];
        y += s_acc[w][d0 + 1] * w_scale[w];
      }
      if (nsplits == 1) {
        out_row[d0] = __float2half(x * inv);
        out_row[d0 + 1] = __float2half(y * inv);
      } else {
        pacc_row[d0] = x;
        pacc_row[d0 + 1] = y;
      }
    }
  }
}

// Second pass: combine the per-split softmax states into the final row.
// out = sum_s acc_s * exp(m_s - max_s m) / sum_s l_s * exp(m_s - max_s m)
template <int HEAD_DIM>
__global__ void decode_attention_merge_kernel(
    const float* __restrict__ p_m, const float* __restrict__ p_l,
    const float* __restrict__ p_acc, __half* __restrict__ out,
    int H, int nsplits) {
  const int head = blockIdx.x;
  const int batch = blockIdx.y;
  const long long base = (static_cast<long long>(batch) * H + head) * nsplits;

  float g_max = -INFINITY;
  for (int s = 0; s < nsplits; ++s) {
    if (p_l[base + s] > 0.f) g_max = fmaxf(g_max, p_m[base + s]);
  }

  __half* out_row = out + (static_cast<long long>(batch) * H + head) * HEAD_DIM;

  if (!(g_max > -INFINITY)) {  // every split empty
    for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x) out_row[d] = __float2half(0.f);
    return;
  }

  float denom = 0.f;
  for (int s = 0; s < nsplits; ++s) {
    const float l = p_l[base + s];
    if (l > 0.f) denom += l * __expf(p_m[base + s] - g_max);
  }
  const float inv = 1.f / denom;

  for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x) {
    float acc = 0.f;
    for (int s = 0; s < nsplits; ++s) {
      const float l = p_l[base + s];
      if (l > 0.f) acc += p_acc[(base + s) * HEAD_DIM + d] * __expf(p_m[base + s] - g_max);
    }
    out_row[d] = __float2half(acc * inv);
  }
}

}  // namespace

// Enough blocks to keep every SM busy, but never so many that a split holds too
// few tokens to amortise its own launch and merge.
//
// kBlocksPerSm is measured, not assumed. ncu on an L4 at batch 8 / ctx 4096
// showed 36.37% achieved occupancy with a 256-block grid -- 4.4 blocks/SM x 4
// warps / 48 max warps = 36.8%, i.e. the limiter was grid size, not the 51
// registers/thread this kernel uses. Sweeping the split count at that shape:
//
//   splits  1      2      4      8      16
//   p50 ms  0.3953 0.2847 0.3154 0.2929 0.3052
//   % peak  56.6   78.6   70.9   76.4   73.3
//
// 2 splits (512 blocks, ~8.8/SM) was best. At batch 32 the grid is already 1024
// blocks and every extra split cost time (53.7 -> 51.7 -> 49.5%), so the target
// is a total block count rather than a batch-size threshold.
int decode_attention_num_splits(int B, int H, int seq_len, int num_sms) {
  constexpr int kMinTokensPerSplit = 256;
  constexpr int kMaxSplits = 32;
  constexpr int kBlocksPerSm = 8;
  const int target = kBlocksPerSm * num_sms;
  const int blocks = B * H;
  if (blocks >= target) return 1;
  int want = (target + blocks - 1) / blocks;
  const int by_len = (seq_len + kMinTokensPerSplit - 1) / kMinTokensPerSplit;
  want = min(want, max(1, by_len));
  return max(1, min(want, kMaxSplits));
}

// Dispatch. HEAD_DIM is a template parameter because it fixes the register
// layout; only the shapes this project actually runs are instantiated.
#define LAUNCH(HD, W, PAGED, BLK)                                             \
  decode_attention_kernel<HD, W, PAGED, BLK>                                   \
      <<<grid, W * kWarpSize, 0, stream>>>(                                    \
          q, k_cache, v_cache, lens, out, p_m, p_l, p_acc, block_table,        \
          max_blocks, H, Hkv, S_max, scale, nsplits);                          \
  if (nsplits > 1) {                                                           \
    decode_attention_merge_kernel<HD><<<dim3(H, B), 128, 0, stream>>>(         \
        p_m, p_l, p_acc, out, H, nsplits);                                     \
  }

// Block size is a template parameter because it fixes the index arithmetic, so
// each supported value is a separate instantiation. Only the paged path varies
// it; the contiguous path never divides by it.
#define DISPATCH_BLOCK(HD, W)                        \
  switch (block_size) {                              \
    case 16:  LAUNCH(HD, W, true, 16);  break;       \
    case 32:  LAUNCH(HD, W, true, 32);  break;       \
    case 64:  LAUNCH(HD, W, true, 64);  break;       \
    case 128: LAUNCH(HD, W, true, 128); break;       \
    default: return cudaErrorInvalidValue;           \
  }

// block_table == nullptr selects the contiguous cache; otherwise the pool is
// addressed through the table with `block_size` tokens per block.
cudaError_t decode_attention_launch(const __half* q, const __half* k_cache,
                                    const __half* v_cache, const int* lens,
                                    __half* out, float* p_m, float* p_l,
                                    float* p_acc, const int* block_table,
                                    int max_blocks, int block_size, int B, int H,
                                    int Hkv, int S_max, int head_dim, float scale,
                                    int nsplits, cudaStream_t stream) {
  dim3 grid(H, B, nsplits);
  constexpr int kWarps = 4;
  if (block_table != nullptr) {
    switch (head_dim) {
      case 64: DISPATCH_BLOCK(64, kWarps); break;
      case 128: DISPATCH_BLOCK(128, kWarps); break;
      default: return cudaErrorInvalidValue;
    }
  } else {
    switch (head_dim) {
      case 64: LAUNCH(64, kWarps, false, 16); break;
      case 128: LAUNCH(128, kWarps, false, 16); break;
      default: return cudaErrorInvalidValue;
    }
  }
  return cudaGetLastError();
}
#undef DISPATCH_BLOCK
#undef LAUNCH
