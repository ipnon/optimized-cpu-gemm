#include <algorithm>

#include "gemm.hpp"

// Block size chosen to fit 3 tiles into L1 cache.
// 3 tiles x (32x32 floats) x 4 bytes = 12KB << 128KB L1
constexpr size_t block_size = 32;  // Tune this (32-64 for M2's 128KB L1)

void gemm_blocked(const float* A, const float* B, float* C, size_t M, size_t N,
                  size_t K) {
  for (size_t i = 0; i < M * N; i++) C[i] = 0;  // Zero C first

  // Iterate over tile positions
  for (size_t i0 = 0; i0 < M; i0 += block_size) {
    // k0 precedes j0 to keep A[i0, k0] tile in cache while scanning all of j0
    // This improves 1024x1024 GEMM from ~2.8 GFLOPs to ~3.2.
    for (size_t k0 = 0; k0 < K; k0 += block_size) {
      for (size_t j0 = 0; j0 < N; j0 += block_size) {
        // Compute one tile multiplication
        // All of these computations fit into L1 because of the block size we
        // chose.
        for (size_t i = i0; i < std::min(i0 + block_size, M); i++) {
          for (size_t j = j0; j < std::min(j0 + block_size, N); j++) {
            // This is a partial dot product.
            // The full C accumulation occurs across multiple shared dimension
            // (k0) iterations.
            for (size_t k = k0; k < std::min(k0 + block_size, K); k++) {
              C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
          }
        }
      }
    }
  }
}
