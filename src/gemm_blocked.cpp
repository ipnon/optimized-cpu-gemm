#include <algorithm>

#include "gemm.hpp"

constexpr size_t block_size = 32;  // Tune this (32-64 for M2's 128KB L1)

void gemm_blocked(const float* A, const float* B, float* C, size_t M, size_t N,
                  size_t K) {
  // Zero C first
  for (size_t i = 0; i < M * N; i++) C[i] = 0;

  // Block over all three dimensions
  for (size_t i0 = 0; i0 < M; i0 += block_size) {
    for (size_t j0 = 0; j0 < N; j0 += block_size) {
      for (size_t k0 = 0; k0 < K; k0 += block_size) {
        // Multiply blocks
        for (size_t i = i0; i < std::min(i0 + block_size, M); i++) {
          for (size_t j = j0; j < std::min(j0 + block_size, N); j++) {
            for (size_t k = k0; k < std::min(k0 + block_size, K); k++) {
              C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
          }
        }
      }
    }
  }
}
