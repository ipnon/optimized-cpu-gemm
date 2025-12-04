#pragma once
#include <chrono>
#include <cstddef>

template <typename GemmFn>
double benchmark_gemm(GemmFn fn, const float* A, const float* B, float* C,
                      size_t M, size_t N, size_t K, int runs = 10) {
  // Warmup
  fn(A, B, C, M, N, K);

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < runs; i++) {
    fn(A, B, C, M, N, K);
  }
  auto end = std::chrono::high_resolution_clock::now();

  double seconds = std::chrono::duration<double>(end - start).count() / runs;
  double flops = 2.0 * M * N * K;
  return flops / seconds / 1e9;
}
