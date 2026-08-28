#!/usr/bin/env bash
# Nsight Compute profile of the fused decode kernel.
#
# Captures the three metrics that decide whether a bandwidth-bound kernel is
# actually good: DRAM throughput (are we near the roofline), achieved occupancy
# (do we have enough warps in flight to hide memory latency), and warp stall
# reasons (if we are off the roofline, what is the stall actually waiting on --
# long_scoreboard means memory, which is expected here; anything else is a bug).
set -euo pipefail

BATCH="${1:-8}"
CTX="${2:-4096}"
OUT="${3:-ncu_decode_attention}"

cd "$(dirname "$0")/.."

if ! command -v ncu >/dev/null 2>&1; then
  echo "ncu not found. On brev/GCP images it ships with the CUDA toolkit:" >&2
  echo "  export PATH=/usr/local/cuda/bin:\$PATH" >&2
  exit 1
fi

# Profile only the fused kernel, skipping the SDPA/flash rows and the JIT build.
cat > /tmp/_ncu_target.py <<PY
import torch
from custom_attn import decode_attention
B, S, H, HKV, D = ${BATCH}, ${CTX}, 32, 8, 64
q = torch.randn(B, H, D, device="cuda", dtype=torch.float16).contiguous()
k = torch.randn(B, HKV, S, D, device="cuda", dtype=torch.float16).contiguous()
v = torch.randn(B, HKV, S, D, device="cuda", dtype=torch.float16).contiguous()
lens = torch.full((B,), S, device="cuda", dtype=torch.int32)
for _ in range(10):            # warm up + trigger the JIT build outside the profile
    decode_attention(q, k, v, lens)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
decode_attention(q, k, v, lens)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
PY

ncu --profile-from-start off \
    --kernel-name-base demangled \
    --kernel-name regex:decode_attention_kernel \
    --set full \
    --export "${OUT}" --force-overwrite \
    python3 /tmp/_ncu_target.py

echo
echo "=== summary (batch=${BATCH}, ctx=${CTX}) ==="
ncu --import "${OUT}.ncu-rep" --page details \
  | grep -E "DRAM Throughput|Memory Throughput|Achieved Occupancy|Theoretical Occupancy|Registers Per Thread|Block Limit|Duration|Compute \(SM\)" || true

echo
echo "=== top warp stall reasons ==="
ncu --import "${OUT}.ncu-rep" --page raw \
  | grep -E "smsp__pcsamp_warps_issue_stalled" | sort -k2 -nr | head -8 || true

echo
echo "wrote ${OUT}.ncu-rep -- open in Nsight Compute UI for the README screenshot"
