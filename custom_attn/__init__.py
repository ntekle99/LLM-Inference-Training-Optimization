"""Fused decode-attention CUDA kernel, with a PyTorch reference for testing.

Dev iteration uses torch.utils.cpp_extension.load (JIT, cached under
TORCH_EXTENSIONS_DIR). The real build is `pip install .` -> setup.py.
"""

import os
import math

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_CSRC = os.path.join(os.path.dirname(_HERE), "csrc")

_ext = None


def _load_ext():
    """JIT-compile on first use. Import stays cheap and CPU-only machines can
    still import this module for the reference path."""
    global _ext
    if _ext is not None:
        return _ext
    try:
        import custom_attn_C as _prebuilt  # installed via setup.py
        _ext = _prebuilt
        return _ext
    except ImportError:
        pass

    from torch.utils.cpp_extension import load

    _ext = load(
        name="custom_attn_C",
        sources=[
            os.path.join(_CSRC, "bindings.cpp"),
            os.path.join(_CSRC, "decode_attention.cu"),
        ],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        verbose=False,
    )
    return _ext


def paged_decode_attention(q, k_pool, v_pool, block_table, lens, scale=None):
    """Fused single-query attention over a paged KV cache.

    q           [B, H, D]                        fp16, contiguous
    k_pool      [num_blocks, Hkv, BLOCK, D]      fp16, contiguous
    v_pool      [num_blocks, Hkv, BLOCK, D]      fp16, contiguous
    block_table [B, max_blocks]                  int32 CUDA
    lens        [B]                              int32 CUDA
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.size(-1))
    return _load_ext().paged_decode_attention(q, k_pool, v_pool, block_table,
                                              lens, float(scale))


def decode_attention(q, k_cache, v_cache, lens, scale=None):
    """Fused single-query attention over a contiguous KV cache.

    q       [B, H, D]           fp16, contiguous  -- the token being generated
    k_cache [B, Hkv, S_max, D]  fp16, contiguous
    v_cache [B, Hkv, S_max, D]  fp16, contiguous
    lens    [B]                 int32 CUDA        -- valid cached tokens per seq
    scale   float, defaults to 1/sqrt(D)

    returns [B, H, D] fp16
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.size(-1))
    return _load_ext().decode_attention(q, k_cache, v_cache, lens, float(scale))


def decode_attention_reference(q, k_cache, v_cache, lens, scale=None):
    """SDPA-based reference. Deliberately independent of the kernel: it builds a
    mask and calls torch's own attention rather than reimplementing softmax, so
    a shared bug in my softmax cannot make the test pass."""
    B, H, D = q.shape
    Hkv, S_max = k_cache.size(1), k_cache.size(2)
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    # GQA: expand each KV head to the query heads that share it.
    rep = H // Hkv
    k = k_cache.repeat_interleave(rep, dim=1)  # [B, H, S_max, D]
    v = v_cache.repeat_interleave(rep, dim=1)

    # Mask out padding beyond each sequence's true length.
    pos = torch.arange(S_max, device=q.device).view(1, 1, 1, S_max)
    valid = pos < lens.to(q.device).view(B, 1, 1, 1)
    mask = torch.zeros(B, 1, 1, S_max, device=q.device, dtype=q.dtype)
    mask.masked_fill_(~valid, float("-inf"))

    out = torch.nn.functional.scaled_dot_product_attention(
        q.unsqueeze(2), k, v, attn_mask=mask, scale=scale
    )  # [B, H, 1, D]
    out = out.squeeze(2)

    # SDPA yields NaN for an all-masked row; the kernel defines empty as zero.
    return torch.where(lens.view(B, 1, 1).to(q.device) > 0, out, torch.zeros_like(out))


from .paged import (PagedKVCache, OutOfBlocks, BLOCK_SIZE,  # noqa: E402
                    SUPPORTED_BLOCK_SIZES)

__all__ = ["decode_attention", "decode_attention_reference",
           "paged_decode_attention", "PagedKVCache", "OutOfBlocks", "BLOCK_SIZE",
           "SUPPORTED_BLOCK_SIZES"]
