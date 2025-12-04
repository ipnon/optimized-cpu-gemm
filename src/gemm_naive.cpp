#include "gemm.hpp"

void gemm_naive(const float* A, const float* B, float* C, size_t M, size_t N,
                size_t K) {
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      C[i * N + j] = 0;
      for (size_t k = 0; k < K; ++k) {
        C[i * N + j] += A[i * K + k] * B[k * N + j];
      }
    }
  }
}
