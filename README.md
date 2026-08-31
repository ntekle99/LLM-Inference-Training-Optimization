# Efficient llm inference + fine-tuning

i wanted to see how far i could push a 1B-parameter model on one GPU without just throwing more hardware at it.

this repo has:

- a fused CUDA decode-attention kernel
- split-K decoding for small batches
- a paged KV cache with prefix sharing
- LoRA fine-tuning with AMP, gradient accumulation, and checkpointing

the model is Meta LLaMA 3.2 1B. the point was to make inference fast and fine-tuning practical on limited hardware.

## quick results

- fused decode attention hits **82.1% of peak DRAM bandwidth** on an NVIDIA L4
- up to **9.98x faster than PyTorch SDPA**
- paged KV caching fits up to **7.98x more short sequences** in the same VRAM budget
- LoRA reduces trainable parameters to **0.2%** of full fine-tuning
- mixed precision cuts memory about in half and speeds up training by about **1.8x**

## fused decode-attention kernel

i wrote an FP16 CUDA kernel for decode attention: generating the next token from an existing KV cache.

the main idea is pretty simple: decode attention is bandwidth-bound. an unfused implementation writes attention scores to memory, reads them back for softmax, writes again, then reads again for the value projection. this kernel keeps the softmax state and accumulation in registers, so K and V are each streamed from memory once.

the benchmark setup:

- NVIDIA L4, 24 GB, sm_89
- CUDA 12.4
- PyTorch 2.6.0 + cu124
- FP16
- LLaMA 3.2 1B attention shape: 32 query heads, 8 KV heads, head dimension 64

| batch | context | implementation | p50 (ms) | p99 (ms) | GB/s | % peak | vs. SDPA |
|---:|---:|---|---:|---:|---:|---:|---:|
| 8 | 4096 | **fused** | **0.2724** | 0.4054 | **246.4** | **82.1%** | **9.98x** |
| 8 | 4096 | FlashAttention-2 | 0.3011 | 0.3932 | 222.9 | 74.3% | 9.03x |
| 8 | 4096 | PyTorch SDPA | 2.7177 | 2.7986 | 24.7 | 8.2% | 1.00x |
| 16 | 4096 | **fused** | **0.7086** | 0.8428 | **189.4** | **63.1%** | **7.32x** |
| 16 | 4096 | FlashAttention-2 | 0.7291 | 0.7721 | 184.1 | 61.4% | 7.12x |
| 32 | 4096 | fused | 1.6077 | 1.7336 | 167.0 | 55.7% | 6.16x |
| 32 | 4096 | **FlashAttention-2** | **1.3005** | 1.3210 | **206.4** | **68.8%** | **7.61x** |
| 32 | 1024 | **fused** | **0.3185** | 0.4413 | **210.7** | **70.2%** | **8.02x** |
| 32 | 1024 | FlashAttention-2 | 0.3420 | 0.4157 | 196.2 | 65.4% | 7.47x |

the kernel beats FlashAttention-2 at batch 8 / context 4096 and batch 32 / context 1024. FlashAttention-2 wins at batch 32 / context 4096. both are much faster than SDPA.

### kernel design

- one block per `(batch, head, split)` with four warps
- Q loads once into registers
- K and V use `half2` loads for coalesced memory access
- each warp keeps a running online-softmax state in registers
- partial softmax states merge using the FlashAttention rescaling identity
- accumulation is FP32; the final output store is FP16
- supports grouped-query attention
- uses split-K / FlashDecoding when the batch is too small to fill the GPU

split-K matters because a normal decode kernel only launches `batch × heads` blocks. at batch 1, that is 32 blocks on a 58-SM L4, so most of the GPU is doing nothing. splitting the sequence across blocks fixes that.

at batch 1 / context 4096, split-K improved latency from **0.1843 ms to 0.0614 ms**: a 3x win from actually filling the machine.

### correctness

- `tests/test_decode_attention.py`: **77 tests passing** against PyTorch SDPA
- `csrc/test_standalone.cu`: 26 cases against a float64 CPU reference
- max absolute error: **2.4e-4**, well inside the `1e-2` gate

the tests cover the annoying cases too: ragged batches, sequence lengths around tile boundaries, garbage past the valid sequence length, and numerically unstable score distributions.

## paged KV cache

a contiguous KV cache preallocates space for every sequence at `max_seq_len`, even when most sequences are short. this cache allocates fixed-size blocks as sequences grow, so it only uses the memory the sequence actually needs.

it also supports prefix sharing: forked sequences point at the same prompt blocks until one writes to them, then copy-on-write kicks in.

### concurrency with an 8 GiB KV-cache budget

| sequence distribution | mean length | contiguous | paged | gain |
|---|---:|---:|---:|---:|
| short-tailed lognormal | 506 | 1024 | **8167** | **7.98x** |
| uniform(1, 4096) | 2072 | 1024 | 2015 | 1.97x |
| all sequences at max length | 4096 | 1024 | 1024 | 1.00x |

if every sequence is actually max length, paging does nothing. if requests are mostly short, it makes a huge difference.

### paging overhead

| batch | context | contiguous p50 | paged p50 | overhead |
|---:|---:|---:|---:|---:|
| 8 | 1024 | 0.0645 ms | 0.0626 ms | ~0% |
| 8 | 4096 | 0.2755 ms | 0.3390 ms | +23% |
| 32 | 1024 | 0.2836 ms | 0.3570 ms | +26% |

paging costs about a quarter of latency at longer contexts. the first version was 2.6x slower, mostly because every K/V load waited on a block-table lookup. hoisting that lookup to one per tile brought the overhead down from +160% to about +23%.

## fine-tuning

fine-tuning uses a 200-example Alpaca subset on an NVIDIA P100 with 16 GB VRAM.

- optimizer: SGD, learning rate `1e-5`
- gradient accumulation: `8`
- LoRA: `r = 16`, `alpha = 32`, dropout `0.05`

| setup | peak memory | runtime / step | notes |
|---|---:|---:|---|
| FP32 baseline | 14,800 MB | 1.12 s | full fine-tuning |
| + mixed precision | 7,800 MB | 0.61 s | about 2x faster |
| + checkpointing | 6,200 MB | 0.73 s | saves activation memory |
| + LoRA | 6,100 MB | 0.68 s | 0.2% of parameters trainable |

## running it

```bash
pip install -e .

python -m pytest tests/ -q
python bench/bench_decode_attention.py --json out.json
python bench/ab.py
bash bench/profile_ncu.sh 8 4096

python -m pytest tests/test_paged_attention.py -q
python bench/bench_paged.py --budget-gb 8
