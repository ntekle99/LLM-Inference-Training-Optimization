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

## ⚙️ System Architecture
efficient-llm/
├── model/
│ ├── llama.py # Transformer core
│ ├── lora.py # Custom LoRA linear modules
│
├── scripts/
│ ├── inference.py # KV cache experiments
│ ├── finetuning.py # Instruction tuning workflow
│ ├── benchmark_inference.py# Memory/runtime profiling
│
├── data/
│ └── alpaca_subset.json # 200-sample instruction dataset
└── README.md

## ⚡ Inference Benchmarks

| **Batch** | **Cache** | **Peak Memory (GB)** | **Runtime (s)** | **Δ Runtime** |
|-----------:|:----------:|:--------------------:|:---------------:|:-------------:|
| 1 | ✅ ON | 3.07 | 0.37 | — |
| 1 | ❌ OFF | 3.23 | 0.41 | +11 % |
| 8 | ✅ ON | 4.50 | 0.52 | — |
| 8 | ❌ OFF | 5.76 | 1.80 | +246 % |
| 16 | ✅ ON | 6.13 | 0.63 | — |
| 16 | ❌ OFF | 8.64 | 4.25 | +574 % |

> **Insight:** KV caching is essential for scalable inference—without it, attention recomputation scales linearly with prompt length × batch size.

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
**Languages:** Python, CUDA  
**Frameworks:** PyTorch (AMP, Checkpointing), LoRA (PEFT)  
**Hardware:** NVIDIA P100 (16 GB)  
**Dataset:** Stanford Alpaca subset  
**Model:** Meta LLaMA 3.2 1B (decoder-only transformer)  

## 📈 Future Directions
- Extend to **quantization-aware training (QAT)** for 4-bit fine-tuning  
- Profile **attention kernel fusion** and **Flash-Attention 2** on larger LLaMA variants  
- Integrate **RLHF or DPO** for alignment-style fine-tuning  
- Build a **web demo** for side-by-side inference comparison (cached vs non-cached)
