// Standalone correctness check: nvcc only, no PyTorch.
//
// The pytest gate in tests/ is the real one -- it compares against torch SDPA.
// This exists so the kernel can be validated the moment a CUDA toolkit is
// present, without waiting on a torch install, and so a kernel bug can be
// isolated from a binding bug.
//
//   nvcc -O3 -arch=sm_89 csrc/test_standalone.cu csrc/decode_attention.cu -o /tmp/t && /tmp/t

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

cudaError_t decode_attention_launch(const __half* q, const __half* k_cache,
                                    const __half* v_cache, const int* lens,
                                    __half* out, float* p_m, float* p_l,
                                    float* p_acc, const int* block_table,
                                    int max_blocks, int block_size, int B, int H,
                                    int Hkv, int S_max, int head_dim, float scale,
                                    int nsplits, cudaStream_t stream);

int decode_attention_num_splits(int B, int H, int seq_len, int num_sms);

#define CHECK(x)                                                          \
  do {                                                                    \
    cudaError_t e = (x);                                                  \
    if (e != cudaSuccess) {                                               \
      printf("CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__); \
      exit(1);                                                            \
    }                                                                     \
  } while (0)

// Double-precision CPU reference with a max-subtracted softmax.
static void reference(const std::vector<float>& q, const std::vector<float>& k,
                      const std::vector<float>& v, const std::vector<int>& lens,
                      std::vector<float>& out, int B, int H, int Hkv, int S_max,
                      int D, float scale) {
  const int rep = H / Hkv;
  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < H; ++h) {
      const int kvh = h / rep;
      const int n = lens[b];
      float* o = &out[(b * H + h) * D];
      for (int d = 0; d < D; ++d) o[d] = 0.f;
      if (n <= 0) continue;

      std::vector<double> score(n);
      double m = -INFINITY;
      for (int t = 0; t < n; ++t) {
        double dot = 0.0;
        for (int d = 0; d < D; ++d) {
          dot += (double)q[(b * H + h) * D + d] *
                 (double)k[((b * Hkv + kvh) * S_max + t) * D + d];
        }
        score[t] = dot * scale;
        if (score[t] > m) m = score[t];
      }
      double sum = 0.0;
      for (int t = 0; t < n; ++t) { score[t] = exp(score[t] - m); sum += score[t]; }
      for (int t = 0; t < n; ++t) {
        const double p = score[t] / sum;
        for (int d = 0; d < D; ++d) {
          o[d] += (float)(p * (double)v[((b * Hkv + kvh) * S_max + t) * D + d]);
        }
      }
    }
  }
}

static float frand(float spread) {
  return spread * (2.f * (float)rand() / (float)RAND_MAX - 1.f);
}

static bool run_case(const char* label, int B, int H, int Hkv, int S_max, int D,
                     std::vector<int> lens, float spread) {
  srand(7);
  const float scale = 1.f / sqrtf((float)D);

  std::vector<float> q(B * H * D), k(B * Hkv * S_max * D), v(B * Hkv * S_max * D);
  for (auto& x : q) x = frand(spread);
  for (auto& x : k) x = frand(spread);
  for (auto& x : v) x = frand(1.f);

  // Round through fp16 so the reference sees exactly the kernel's inputs and
  // the reported error is the kernel's, not the input quantisation's.
  std::vector<__half> qh(q.size()), kh(k.size()), vh(v.size());
  for (size_t i = 0; i < q.size(); ++i) { qh[i] = __float2half(q[i]); q[i] = __half2float(qh[i]); }
  for (size_t i = 0; i < k.size(); ++i) { kh[i] = __float2half(k[i]); k[i] = __half2float(kh[i]); }
  for (size_t i = 0; i < v.size(); ++i) { vh[i] = __float2half(v[i]); v[i] = __half2float(vh[i]); }

  __half *dq, *dk, *dv, *dout;
  int* dlens;
  CHECK(cudaMalloc(&dq, qh.size() * sizeof(__half)));
  CHECK(cudaMalloc(&dk, kh.size() * sizeof(__half)));
  CHECK(cudaMalloc(&dv, vh.size() * sizeof(__half)));
  CHECK(cudaMalloc(&dout, qh.size() * sizeof(__half)));
  CHECK(cudaMalloc(&dlens, lens.size() * sizeof(int)));
  CHECK(cudaMemcpy(dq, qh.data(), qh.size() * sizeof(__half), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dk, kh.data(), kh.size() * sizeof(__half), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dv, vh.data(), vh.size() * sizeof(__half), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dlens, lens.data(), lens.size() * sizeof(int), cudaMemcpyHostToDevice));

  int num_sms = 0;
  CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));
  const int nsplits = decode_attention_num_splits(B, H, S_max, num_sms);

  float *pm = nullptr, *pl = nullptr, *pa = nullptr;
  if (nsplits > 1) {
    CHECK(cudaMalloc(&pm, (size_t)B * H * nsplits * sizeof(float)));
    CHECK(cudaMalloc(&pl, (size_t)B * H * nsplits * sizeof(float)));
    CHECK(cudaMalloc(&pa, (size_t)B * H * nsplits * D * sizeof(float)));
  }

  CHECK(decode_attention_launch(dq, dk, dv, dlens, dout, pm, pl, pa, nullptr, 0,
                                16, B, H, Hkv, S_max, D, scale, nsplits, 0));
  CHECK(cudaDeviceSynchronize());
  if (pm) { cudaFree(pm); cudaFree(pl); cudaFree(pa); }

  std::vector<__half> outh(qh.size());
  CHECK(cudaMemcpy(outh.data(), dout, outh.size() * sizeof(__half), cudaMemcpyDeviceToHost));

  std::vector<float> ref(q.size());
  reference(q, k, v, lens, ref, B, H, Hkv, S_max, D, scale);

  float max_err = 0.f;
  bool finite = true;
  for (size_t i = 0; i < ref.size(); ++i) {
    const float got = __half2float(outh[i]);
    if (!isfinite(got)) finite = false;
    max_err = fmaxf(max_err, fabsf(got - ref[i]));
  }

  cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dout); cudaFree(dlens);

  const bool ok = finite && max_err < 1e-2f;
  printf("%-42s B=%-3d H=%-3d Hkv=%-3d S=%-5d D=%-4d splits=%-3d max_err=%.5f  %s\n",
         label, B, H, Hkv, S_max, D, nsplits, max_err,
         ok ? "PASS" : (finite ? "FAIL" : "FAIL(nan)"));
  return ok;
}

int main() {
  int fails = 0;

  fails += !run_case("llama-3.2-1B geometry, ctx 4096", 8, 32, 8, 4096, 64, std::vector<int>(8, 4096), 1.f);
  fails += !run_case("llama geometry, ctx 256", 1, 32, 8, 256, 64, std::vector<int>(1, 256), 1.f);
  fails += !run_case("head_dim 128", 4, 8, 8, 512, 128, std::vector<int>(4, 512), 1.f);
  fails += !run_case("no grouping (H==Hkv)", 4, 8, 8, 1024, 64, std::vector<int>(4, 1024), 1.f);
  fails += !run_case("max grouping (Hkv==1)", 4, 8, 1, 1024, 64, std::vector<int>(4, 1024), 1.f);

  // Tiling boundaries: 4 warps x 4 tokens = 16 tokens per full iteration.
  for (int n : {1, 2, 3, 4, 5, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 129}) {
    char buf[64];
    snprintf(buf, sizeof(buf), "tiling boundary len=%d", n);
    fails += !run_case(buf, 2, 8, 8, 256, 64, std::vector<int>(2, n), 1.f);
  }

  fails += !run_case("ragged lengths in batch", 8, 32, 8, 4096, 64,
                     {4096, 1, 333, 17, 2048, 0, 64, 65}, 1.f);
  fails += !run_case("empty cache -> zeros", 2, 8, 8, 64, 64, {0, 0}, 1.f);

  // Online softmax must keep every exp() argument <= 0; a naive exp(score)
  // overflows fp16 range here and returns NaN.
  for (float s : {4.f, 8.f, 16.f}) {
    char buf[64];
    snprintf(buf, sizeof(buf), "overflow numerics spread=%.0f", s);
    fails += !run_case(buf, 2, 8, 8, 1024, 64, std::vector<int>(2, 1024), s);
  }

  printf("\n%s (%d failing cases)\n", fails ? "FAILED" : "ALL PASS", fails);
  return fails ? 1 : 0;
}
