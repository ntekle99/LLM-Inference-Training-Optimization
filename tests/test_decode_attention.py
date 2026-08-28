"""Correctness gate for the fused decode-attention kernel.

Gate (from the spec): max abs error vs torch SDPA < 1e-2 in FP16, across a
matrix of batch sizes, head counts and sequence lengths.

The interesting tests are not the happy path -- they are the ones aimed at where
a fused online-softmax kernel actually breaks: sequence lengths that do not
divide the warp tiling, ragged lengths within a batch, and score distributions
that overflow a naive (non-online) softmax.
"""

import math

import pytest
import torch

from custom_attn import decode_attention, decode_attention_reference

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fused decode attention requires CUDA"
)

TOL = 1e-2  # the spec's gate

# LLaMA-3.2-1B is H=32, Hkv=8, D=64. The rest cover the GQA edge cases:
# no grouping (H==Hkv) and maximal grouping (Hkv==1).
HEAD_CONFIGS = [(32, 8), (8, 8), (8, 1), (32, 32)]


def _mk(B, H, Hkv, S_max, D, lens, dtype=torch.float16, spread=1.0):
    g = torch.Generator(device="cuda").manual_seed(1234)
    q = (torch.randn(B, H, D, generator=g, device="cuda", dtype=torch.float32) * spread).to(dtype)
    k = (torch.randn(B, Hkv, S_max, D, generator=g, device="cuda", dtype=torch.float32) * spread).to(dtype)
    v = torch.randn(B, Hkv, S_max, D, generator=g, device="cuda", dtype=torch.float32).to(dtype)
    lens_t = torch.tensor(lens, device="cuda", dtype=torch.int32)
    return q.contiguous(), k.contiguous(), v.contiguous(), lens_t


def _max_err(got, want):
    return (got.float() - want.float()).abs().max().item()


def _check(B, H, Hkv, S_max, D, lens, spread=1.0):
    q, k, v, lens_t = _mk(B, H, Hkv, S_max, D, lens, spread=spread)
    got = decode_attention(q, k, v, lens_t)
    want = decode_attention_reference(q, k, v, lens_t)
    assert torch.isfinite(got.float()).all(), "kernel produced NaN/Inf"
    err = _max_err(got, want)
    assert err < TOL, f"max abs err {err:.5f} >= {TOL} (B={B},H={H},Hkv={Hkv},S={S_max},D={D})"
    return err


# ---------------------------------------------------------------- the matrix

@pytest.mark.parametrize("B", [1, 4, 8, 16])
@pytest.mark.parametrize("H,Hkv", HEAD_CONFIGS)
@pytest.mark.parametrize("S", [256, 1024, 4096])
def test_matrix_full_length(B, H, Hkv, S):
    """The headline matrix: every sequence uses the whole cache."""
    _check(B, H, Hkv, S, 64, [S] * B)


@pytest.mark.parametrize("D", [64, 128])
def test_head_dims(D):
    _check(4, 8, 8, 512, D, [512] * 4)


# --------------------------------------------------- tiling boundary lengths

# A warp handles kUnroll=4 tokens per step and there are 4 warps, so the kernel
# processes 16 tokens per full iteration. These lengths straddle that boundary:
# off-by-one errors in the tail-handling show up here and nowhere else.
@pytest.mark.parametrize("length", [1, 2, 3, 4, 5, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 129])
def test_tiling_boundaries(length):
    _check(2, 8, 8, 256, 64, [length, length])


def test_length_one():
    """With a single cached token, softmax is exactly 1.0 and the output must be
    exactly that token's V row -- a case where the reference cannot hide an error."""
    q, k, v, lens_t = _mk(2, 8, 8, 64, 64, [1, 1])
    got = decode_attention(q, k, v, lens_t)
    assert _max_err(got, v[:, :, 0, :]) < TOL


def test_zero_length_is_zero():
    """Empty cache is defined as a zero output, not NaN. A warp that sees no
    tokens carries run_max = -inf; the merge must not turn that into NaN."""
    q, k, v, lens_t = _mk(2, 8, 8, 64, 64, [0, 0])
    got = decode_attention(q, k, v, lens_t)
    assert torch.isfinite(got.float()).all()
    assert got.float().abs().max().item() == 0.0


# -------------------------------------------------------------- raggedness

def test_ragged_lengths_in_batch():
    """Different lengths per sequence. If the kernel read `lens` for the wrong
    batch index, uniform-length tests would all still pass."""
    lens = [4096, 1, 333, 17, 2048, 0, 64, 65]
    _check(8, 32, 8, 4096, 64, lens)


def test_padding_beyond_len_is_ignored():
    """Garbage past each sequence length must not change the result -- this is
    what proves the length bound is real and not just arithmetically harmless."""
    B, H, Hkv, S, D = 4, 8, 8, 512, 64
    lens = [100, 7, 512, 33]
    q, k, v, lens_t = _mk(B, H, Hkv, S, D, lens)
    clean = decode_attention(q, k, v, lens_t)

    k2, v2 = k.clone(), v.clone()
    for b, n in enumerate(lens):
        k2[b, :, n:, :] = 100.0   # would dominate softmax if ever read
        v2[b, :, n:, :] = -50.0
    dirty = decode_attention(q, k2, v2, lens_t)
    assert _max_err(clean, dirty) == 0.0, "kernel read past seq_len"


# --------------------------------------------------------------- numerics

@pytest.mark.parametrize("spread", [4.0, 8.0, 16.0])
def test_online_softmax_does_not_overflow(spread):
    """Large Q/K magnitudes push raw scores far above the FP16 max (65504). The
    online rescaling must keep every exp() argument <= 0. A kernel that computed
    exp(score) directly returns NaN here while passing every test above."""
    err = _check(2, 8, 8, 1024, 64, [1024, 1024], spread=spread)
    assert err < TOL


def test_uniform_scores_are_a_plain_mean():
    """Identical K rows give a uniform softmax, so the output is the mean of V.
    This checks the running-sum normalisation independently of the reference."""
    B, H, S, D = 2, 8, 300, 64
    q = torch.randn(B, H, D, device="cuda", dtype=torch.float16).contiguous()
    k = torch.ones(B, H, S, D, device="cuda", dtype=torch.float16).contiguous()
    v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float32)
    lens_t = torch.tensor([S] * B, device="cuda", dtype=torch.int32)
    got = decode_attention(q, k.contiguous(), v.half().contiguous(), lens_t)
    assert _max_err(got, v.mean(dim=2)) < TOL


def test_gqa_head_mapping():
    """Query heads sharing a KV head must agree when they carry the same Q. If
    the kv_head index were computed as `head % Hkv` instead of `head / (H/Hkv)`,
    the grouping would be wrong while shapes stayed valid."""
    B, H, Hkv, S, D = 1, 32, 8, 128, 64
    rep = H // Hkv
    q1 = torch.randn(B, 1, D, device="cuda", dtype=torch.float16)
    q = q1.repeat(1, H, 1).contiguous()
    k = torch.randn(B, Hkv, S, D, device="cuda", dtype=torch.float16).contiguous()
    v = torch.randn(B, Hkv, S, D, device="cuda", dtype=torch.float16).contiguous()
    lens_t = torch.tensor([S], device="cuda", dtype=torch.int32)
    got = decode_attention(q, k, v, lens_t)
    for g in range(Hkv):
        grp = got[0, g * rep:(g + 1) * rep, :]
        assert _max_err(grp, grp[0:1].expand_as(grp)) < 1e-3, f"group {g} disagrees"


def test_determinism():
    """Same inputs, same output, bit for bit. Cross-warp merge order is fixed,
    so there is no excuse for run-to-run drift."""
    q, k, v, lens_t = _mk(4, 32, 8, 1024, 64, [1024, 500, 1, 999])
    a = decode_attention(q, k, v, lens_t)
    for _ in range(5):
        assert torch.equal(a, decode_attention(q, k, v, lens_t))


# ------------------------------------------------------------ input guards

def test_rejects_bad_inputs():
    q, k, v, lens_t = _mk(2, 8, 8, 64, 64, [64, 64])
    with pytest.raises(RuntimeError):
        decode_attention(q.float(), k, v, lens_t)             # wrong dtype
    with pytest.raises(RuntimeError):
        decode_attention(q, k, v, lens_t.long())              # wrong lens dtype
    with pytest.raises(RuntimeError):
        decode_attention(q.transpose(0, 1), k, v, lens_t)     # non-contiguous
    with pytest.raises(RuntimeError):
        decode_attention(q.cpu(), k.cpu(), v.cpu(), lens_t.cpu())  # not CUDA
