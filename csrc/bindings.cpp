// PyTorch bindings for the fused decode-attention kernel.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cstdlib>

cudaError_t decode_attention_launch(const __half* q, const __half* k_cache,
                                    const __half* v_cache, const int* lens,
                                    __half* out, float* p_m, float* p_l,
                                    float* p_acc, const int* block_table,
                                    int max_blocks, int block_size, int B, int H,
                                    int Hkv, int S_max, int head_dim, float scale,
                                    int nsplits, cudaStream_t stream);

int decode_attention_num_splits(int B, int H, int seq_len, int num_sms);

namespace {

inline const __half* half_ptr(const at::Tensor& t) {
  return reinterpret_cast<const __half*>(t.data_ptr<at::Half>());
}

void check_kv(const at::Tensor& t, const char* name, int B, int S_max, int D) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.scalar_type() == at::kHalf, name, " must be float16");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(t.dim() == 4, name, " must be [B, Hkv, S_max, D]");
  TORCH_CHECK(t.size(0) == B, name, " batch mismatch");
  TORCH_CHECK(t.size(2) == S_max, name, " S_max mismatch");
  TORCH_CHECK(t.size(3) == D, name, " head_dim mismatch");
}

}  // namespace

// q [B,H,D] fp16 | k_cache,v_cache [B,Hkv,S_max,D] fp16 | lens [B] int32
// -> out [B,H,D] fp16
at::Tensor decode_attention(at::Tensor q, at::Tensor k_cache, at::Tensor v_cache,
                            at::Tensor lens, double scale) {
  TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
  TORCH_CHECK(q.scalar_type() == at::kHalf, "q must be float16");
  TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
  TORCH_CHECK(q.dim() == 3, "q must be [B, H, D]");

  const int B = q.size(0), H = q.size(1), D = q.size(2);
  const int Hkv = k_cache.size(1), S_max = k_cache.size(2);

  check_kv(k_cache, "k_cache", B, S_max, D);
  check_kv(v_cache, "v_cache", B, S_max, D);
  TORCH_CHECK(k_cache.size(1) == v_cache.size(1), "k/v head count mismatch");
  TORCH_CHECK(H % Hkv == 0, "H (", H, ") must be divisible by Hkv (", Hkv, ")");
  TORCH_CHECK(D == 64 || D == 128, "head_dim must be 64 or 128, got ", D);

  TORCH_CHECK(lens.is_cuda() && lens.scalar_type() == at::kInt,
              "lens must be an int32 CUDA tensor");
  TORCH_CHECK(lens.is_contiguous() && lens.dim() == 1 && lens.size(0) == B,
              "lens must be contiguous [B]");

  at::Tensor out = at::empty_like(q);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // The split count is chosen from S_max rather than the true per-sequence
  // lengths: those live on the device, and reading them here would force a
  // host sync on every decode step. S_max is an upper bound, so this can only
  // over-split, which the kernel handles by emitting empty partial states.
  const int num_sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  // CUSTOM_ATTN_SPLITS forces the split count, for sweeping the heuristic
  // against measurement instead of guessing its constants.
  static const int forced = [] {
    const char* e = std::getenv("CUSTOM_ATTN_SPLITS");
    return e ? std::atoi(e) : 0;
  }();
  const int nsplits =
      forced > 0 ? forced : decode_attention_num_splits(B, H, S_max, num_sms);

  at::Tensor p_m, p_l, p_acc;
  float *pm = nullptr, *pl = nullptr, *pa = nullptr;
  if (nsplits > 1) {
    auto opts = q.options().dtype(at::kFloat);
    p_m = at::empty({B, H, nsplits}, opts);
    p_l = at::empty({B, H, nsplits}, opts);
    p_acc = at::empty({B, H, nsplits, D}, opts);
    pm = p_m.data_ptr<float>();
    pl = p_l.data_ptr<float>();
    pa = p_acc.data_ptr<float>();
  }

  cudaError_t err = decode_attention_launch(
      half_ptr(q), half_ptr(k_cache), half_ptr(v_cache), lens.data_ptr<int>(),
      reinterpret_cast<__half*>(out.data_ptr<at::Half>()), pm, pl, pa,
      /*block_table=*/nullptr, /*max_blocks=*/0, /*block_size=*/16, B, H, Hkv,
      S_max, D, static_cast<float>(scale), nsplits, stream);
  TORCH_CHECK(err == cudaSuccess, "decode_attention launch failed: ",
              cudaGetErrorString(err));
  return out;
}

// q [B,H,D] fp16 | k_pool,v_pool [num_blocks, Hkv, BLOCK, D] fp16
// block_table [B, max_blocks] int32 | lens [B] int32 -> out [B,H,D] fp16
at::Tensor paged_decode_attention(at::Tensor q, at::Tensor k_pool,
                                  at::Tensor v_pool, at::Tensor block_table,
                                  at::Tensor lens, double scale) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kHalf && q.is_contiguous(),
              "q must be a contiguous fp16 CUDA tensor");
  TORCH_CHECK(q.dim() == 3, "q must be [B, H, D]");
  const int B = q.size(0), H = q.size(1), D = q.size(2);

  TORCH_CHECK(k_pool.dim() == 4 && v_pool.dim() == 4,
              "pools must be [num_blocks, Hkv, BLOCK, D]");
  TORCH_CHECK(k_pool.sizes() == v_pool.sizes(), "k/v pools must match");
  TORCH_CHECK(k_pool.is_cuda() && k_pool.scalar_type() == at::kHalf && k_pool.is_contiguous(),
              "k_pool must be a contiguous fp16 CUDA tensor");
  TORCH_CHECK(v_pool.is_cuda() && v_pool.scalar_type() == at::kHalf && v_pool.is_contiguous(),
              "v_pool must be a contiguous fp16 CUDA tensor");

  const int Hkv = k_pool.size(1), block = k_pool.size(2);
  TORCH_CHECK(block == 16 || block == 32 || block == 64 || block == 128,
              "block size must be 16, 32, 64 or 128, got ", block);
  TORCH_CHECK(k_pool.size(3) == D, "pool head_dim mismatch");
  TORCH_CHECK(H % Hkv == 0, "H (", H, ") must be divisible by Hkv (", Hkv, ")");
  TORCH_CHECK(D == 64 || D == 128, "head_dim must be 64 or 128, got ", D);

  TORCH_CHECK(block_table.is_cuda() && block_table.scalar_type() == at::kInt &&
                  block_table.is_contiguous() && block_table.dim() == 2 &&
                  block_table.size(0) == B,
              "block_table must be a contiguous int32 CUDA tensor [B, max_blocks]");
  TORCH_CHECK(lens.is_cuda() && lens.scalar_type() == at::kInt &&
                  lens.is_contiguous() && lens.dim() == 1 && lens.size(0) == B,
              "lens must be a contiguous int32 CUDA tensor [B]");

  const int max_blocks = block_table.size(1);
  // S_max only bounds the split heuristic here; the real extent is per-sequence.
  const int S_max = max_blocks * block;

  at::Tensor out = at::empty_like(q);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int num_sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  static const int forced_p = [] {
    const char* e = std::getenv("CUSTOM_ATTN_SPLITS");
    return e ? std::atoi(e) : 0;
  }();
  const int nsplits =
      forced_p > 0 ? forced_p : decode_attention_num_splits(B, H, S_max, num_sms);

  at::Tensor p_m, p_l, p_acc;
  float *pm = nullptr, *pl = nullptr, *pa = nullptr;
  if (nsplits > 1) {
    auto opts = q.options().dtype(at::kFloat);
    p_m = at::empty({B, H, nsplits}, opts);
    p_l = at::empty({B, H, nsplits}, opts);
    p_acc = at::empty({B, H, nsplits, D}, opts);
    pm = p_m.data_ptr<float>();
    pl = p_l.data_ptr<float>();
    pa = p_acc.data_ptr<float>();
  }

  cudaError_t err = decode_attention_launch(
      half_ptr(q), half_ptr(k_pool), half_ptr(v_pool), lens.data_ptr<int>(),
      reinterpret_cast<__half*>(out.data_ptr<at::Half>()), pm, pl, pa,
      block_table.data_ptr<int>(), max_blocks, block, B, H, Hkv, S_max, D,
      static_cast<float>(scale), nsplits, stream);
  TORCH_CHECK(err == cudaSuccess, "paged_decode_attention launch failed: ",
              cudaGetErrorString(err));
  return out;
}

// Exposed so the split heuristic can be inspected from the benchmark rather
// than inferred from timings.
int num_splits_for(int B, int H, int seq_len) {
  const int num_sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  return decode_attention_num_splits(B, H, seq_len, num_sms);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("num_splits_for", &num_splits_for, "Split count the heuristic picks",
        py::arg("B"), py::arg("H"), py::arg("seq_len"));
  m.def("paged_decode_attention", &paged_decode_attention,
        "Fused decode attention over a paged KV cache (fp16)",
        py::arg("q"), py::arg("k_pool"), py::arg("v_pool"),
        py::arg("block_table"), py::arg("lens"), py::arg("scale"));
  m.def("decode_attention", &decode_attention,
        "Fused single-query decode attention (fp16)",
        py::arg("q"), py::arg("k_cache"), py::arg("v_cache"), py::arg("lens"),
        py::arg("scale"));
}
