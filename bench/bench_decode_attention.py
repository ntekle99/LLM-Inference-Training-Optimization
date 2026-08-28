"""Decode-attention latency benchmark: fused kernel vs PyTorch SDPA vs FlashAttention-2.

Reports p50/p99 per-call latency and achieved DRAM bandwidth as a fraction of the
GPU's theoretical peak. For a bandwidth-bound decode kernel that fraction is the
honest efficiency number -- FLOP/s would flatter it, since the kernel does almost
no arithmetic per byte moved.

Every row is tagged with the GPU name and dtype, because a decode-attention
number without a named GPU is not a claim about anything.

  python bench/bench_decode_attention.py
  python bench/bench_decode_attention.py --batches 1,8 --contexts 4096 --json out.json
"""

import argparse
import json
import math
import statistics
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from custom_attn import decode_attention, decode_attention_reference  # noqa: E402

# LLaMA-3.2-1B attention geometry.
N_HEADS, N_KV_HEADS, HEAD_DIM = 32, 8, 64

# Theoretical peak DRAM bandwidth, GB/s (vendor spec). Reporting a percentage
# against a guessed peak would be worse than reporting none, so an unknown GPU
# is an error the user resolves with --peak-bw-gbps rather than a silent default.
PEAK_BW_GBPS = {
    "Tesla P100-PCIE-16GB": 732.0,
    "Tesla P100-SXM2-16GB": 732.0,
    "Tesla V100-SXM2-16GB": 900.0,
    "Tesla T4": 320.0,
    "NVIDIA L4": 300.0,
    "NVIDIA L40S": 864.0,
    "NVIDIA A10G": 600.0,
    "NVIDIA A100-SXM4-40GB": 1555.0,
    "NVIDIA A100-SXM4-80GB": 2039.0,
    "NVIDIA A100-PCIE-40GB": 1555.0,
    "NVIDIA H100 PCIe": 2039.0,
    "NVIDIA H100 80GB HBM3": 3350.0,
}


def time_calls(fn, warmup, iters):
    """Per-call latencies in ms, timed with CUDA events.

    One event pair per iteration rather than one pair around the whole loop:
    percentiles need the distribution, and a loop-level mean would hide exactly
    the tail this benchmark exists to measure.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return sorted(s.elapsed_time(e) for s, e in zip(starts, ends))


def pct(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    idx = min(len(sorted_vals) - 1, max(0, math.ceil(p / 100.0 * len(sorted_vals)) - 1))
    return sorted_vals[idx]


def kv_bytes(B, S):
    """Bytes of K and V that any correct implementation must read: the compulsory
    traffic. Q and the output are ~S times smaller and are ignored."""
    return 2 * B * N_KV_HEADS * S * HEAD_DIM * 2


def l2_cache_bytes():
    try:
        return torch.cuda.get_device_properties(0).L2_cache_size
    except AttributeError:
        return 0


def try_flash_attn():
    try:
        from flash_attn import flash_attn_with_kvcache  # noqa: F401
        return flash_attn_with_kvcache
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="1,8,16,32")
    ap.add_argument("--contexts", default="256,1024,4096")
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--iters", type=int, default=250)
    ap.add_argument("--peak-bw-gbps", type=float, default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("no CUDA device")

    gpu = torch.cuda.get_device_name(0)
    peak = args.peak_bw_gbps or PEAK_BW_GBPS.get(gpu)
    if peak is None:
        sys.exit(
            f"unknown GPU {gpu!r}: pass --peak-bw-gbps <vendor spec> so the "
            f"bandwidth column means something"
        )

    batches = [int(x) for x in args.batches.split(",")]
    contexts = [int(x) for x in args.contexts.split(",")]
    flash = try_flash_attn()

    print(f"GPU:   {gpu}")
    print(f"dtype: float16   heads: {N_HEADS} (kv {N_KV_HEADS})   head_dim: {HEAD_DIM}")
    print(f"peak DRAM bandwidth: {peak:.0f} GB/s   warmup {args.warmup}, iters {args.iters}")
    print(f"FlashAttention-2: {'available' if flash else 'not installed - column omitted'}\n")

    l2 = l2_cache_bytes()
    print(f"L2 cache: {l2/2**20:.0f} MiB -- shapes whose KV working set fits in L2 are")
    print(f"served from cache, so their GB/s is not a DRAM number and is marked (L2).\n")

    hdr = (f"{'batch':>5} {'ctx':>6} {'impl':>10} {'p50 ms':>9} {'p99 ms':>9} "
           f"{'GB/s':>9} {'% peak':>8} {'vs SDPA':>9} {'workset':>9}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for B in batches:
        for S in contexts:
            q = torch.randn(B, N_HEADS, HEAD_DIM, device="cuda", dtype=torch.float16).contiguous()
            k = torch.randn(B, N_KV_HEADS, S, HEAD_DIM, device="cuda", dtype=torch.float16).contiguous()
            v = torch.randn(B, N_KV_HEADS, S, HEAD_DIM, device="cuda", dtype=torch.float16).contiguous()
            lens = torch.full((B,), S, device="cuda", dtype=torch.int32)

            # Correctness is re-checked at every benchmarked shape. A fast wrong
            # kernel is the failure mode this guards against.
            err = (decode_attention(q, k, v, lens).float()
                   - decode_attention_reference(q, k, v, lens).float()).abs().max().item()
            if err >= 1e-2:
                sys.exit(f"correctness gate failed at B={B} S={S}: max abs err {err:.4f}")

            ws = kv_bytes(B, S)
            l2_resident = l2 and ws < l2
            ws_tag = f"{ws/2**20:.0f}M" + ("(L2)" if l2_resident else "")

            impls = {
                "sdpa": lambda: decode_attention_reference(q, k, v, lens),
                "fused": lambda: decode_attention(q, k, v, lens),
            }
            if flash:
                kf = k.transpose(1, 2).contiguous()  # flash wants [B, S, Hkv, D]
                vf = v.transpose(1, 2).contiguous()
                qf = q.unsqueeze(1).contiguous()     # [B, 1, H, D]
                impls["flash2"] = lambda: flash(qf, kf, vf, cache_seqlens=lens)

            sdpa_p50 = None
            for name, fn in impls.items():
                lat = time_calls(fn, args.warmup, args.iters)
                p50, p99 = pct(lat, 50), pct(lat, 99)
                gbps = kv_bytes(B, S) / (p50 * 1e-3) / 1e9
                if name == "sdpa":
                    sdpa_p50 = p50
                speedup = sdpa_p50 / p50 if sdpa_p50 else float("nan")
                pct_str = "     n/a" if l2_resident else f"{100*gbps/peak:>7.1f}%"
                print(f"{B:>5} {S:>6} {name:>10} {p50:>9.4f} {p99:>9.4f} "
                      f"{gbps:>9.1f} {pct_str} {speedup:>8.2f}x {ws_tag:>9}")
                rows.append(dict(gpu=gpu, dtype="float16", batch=B, context=S, impl=name,
                                 p50_ms=p50, p99_ms=p99, mean_ms=statistics.fmean(lat),
                                 achieved_gbps=gbps,
                                 pct_peak=None if l2_resident else 100 * gbps / peak,
                                 kv_working_set_bytes=ws, l2_resident=bool(l2_resident),
                                 speedup_vs_sdpa=speedup, max_abs_err_vs_sdpa=err))
            print()

    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(gpu=gpu, peak_bw_gbps=peak, warmup=args.warmup,
                           iters=args.iters, rows=rows), f, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
