"""Interleaved three-way comparison at a fixed shape.

Why not just read the big sweep table: the same shape measured 0.42ms there and
0.28ms in isolation. The sweep runs minutes of continuous work before reaching
the later rows, and this is a 72W L4 -- sustained load pulls the clocks down, so
absolute numbers drift with position in the table. Ratios inside one table stay
fair (every implementation is measured under the same drift), but a headline
number needs the shapes measured head-to-head with the order rotated.

Rotating the order each round matters: whichever implementation runs first sees
the coolest GPU, so a fixed order silently hands it a win.

Clocks are sampled *during* the timed region by a background thread. Sampling
after torch.cuda.synchronize() -- which is what an earlier version of this file
did -- reads the idle clock the GPU drops to once the work is done, and reports
a throttled run as a fast one.
"""

import math
import os
import statistics
import subprocess
import sys
import threading

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from custom_attn import decode_attention, decode_attention_reference  # noqa: E402

N_HEADS, N_KV_HEADS, HEAD_DIM = 32, 8, 64
SHAPES = [(8, 4096), (16, 4096), (32, 4096), (32, 1024)]
ROUNDS = 3
ITERS = 150
WARMUP = 25
PEAK_GBPS = 300.0  # L4 vendor spec


class ClockSampler(threading.Thread):
    """Median SM clock and max temperature observed while the timing loop runs."""

    def __init__(self):
        super().__init__(daemon=True)
        self.samples, self.temps, self._done = [], [], threading.Event()

    def run(self):
        while not self._done.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout.strip()
                clk, tmp = out.splitlines()[0].split(",")
                self.samples.append(int(clk))
                self.temps.append(int(tmp))
            except Exception:
                pass
            self._done.wait(0.05)

    def stop(self):
        self._done.set()
        self.join(timeout=2)
        return (statistics.median(self.samples) if self.samples else -1,
                max(self.temps) if self.temps else -1)


def time_impl(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    st = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    en = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    sampler = ClockSampler()
    sampler.start()
    for i in range(ITERS):
        st[i].record()
        fn()
        en[i].record()
    torch.cuda.synchronize()
    clk, tmp = sampler.stop()
    lat = sorted(s.elapsed_time(e) for s, e in zip(st, en))
    return lat, clk, tmp


def pct(v, p):
    return v[min(len(v) - 1, max(0, math.ceil(p / 100.0 * len(v)) - 1))]


def main():
    try:
        from flash_attn import flash_attn_with_kvcache as flash
    except Exception:
        flash = None

    gpu = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu}   dtype float16   {ROUNDS} rounds x {ITERS} iters, order rotated\n")

    for B, S in SHAPES:
        q = torch.randn(B, N_HEADS, HEAD_DIM, device="cuda", dtype=torch.float16).contiguous()
        k = torch.randn(B, N_KV_HEADS, S, HEAD_DIM, device="cuda", dtype=torch.float16).contiguous()
        v = torch.randn(B, N_KV_HEADS, S, HEAD_DIM, device="cuda", dtype=torch.float16).contiguous()
        lens = torch.full((B,), S, device="cuda", dtype=torch.int32)
        kv = 2 * B * N_KV_HEADS * S * HEAD_DIM * 2

        impls = {"sdpa": lambda: decode_attention_reference(q, k, v, lens),
                 "fused": lambda: decode_attention(q, k, v, lens)}
        if flash:
            kf, vf = k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous()
            qf = q.unsqueeze(1).contiguous()
            impls["flash2"] = lambda: flash(qf, kf, vf, cache_seqlens=lens)

        names = list(impls)
        acc = {n: [] for n in names}
        meta = {n: [] for n in names}
        for r in range(ROUNDS):
            for n in names[r % len(names):] + names[:r % len(names)]:   # rotate
                lat, clk, tmp = time_impl(impls[n])
                acc[n].append(pct(lat, 50))
                meta[n].append((pct(lat, 99), clk, tmp))

        print(f"batch {B}, ctx {S}   KV working set {kv/2**20:.0f} MiB")
        base = statistics.median(acc["sdpa"])
        for n in names:
            p50 = statistics.median(acc[n])
            p99 = statistics.median(m[0] for m in meta[n])
            gbps = kv / (p50 * 1e-3) / 1e9
            clk = statistics.median(m[1] for m in meta[n])
            tmp = max(m[2] for m in meta[n])
            print(f"  {n:>7}  p50 {p50:7.4f}  p99 {p99:7.4f}  {gbps:6.1f} GB/s "
                  f"({100*gbps/PEAK_GBPS:4.1f}% peak)  {base/p50:5.2f}x  "
                  f"[{clk:.0f}MHz {tmp}C]  runs={[f'{x:.4f}' for x in acc[n]]}")
        print()


if __name__ == "__main__":
    main()
