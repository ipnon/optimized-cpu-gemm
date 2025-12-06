#include "gemm.hpp"

// Row-major indexing: element [row][col] is at address [row * num_cols + col]

void gemm_naive(const float* A, const float* B, float* C, size_t M, size_t N,
                size_t K) {
  for (size_t i = 0; i < M; ++i) {    // For each row of C
    for (size_t j = 0; j < N; ++j) {  // For each column of C
      C[i * N + j] = 0;
      for (size_t k = 0; k < K; ++k) {  // Sum over shared dimension
        // A[i,k]: sequential in k (stride 1) is cache friendly
        // B[k,j]: sequential in k (stride N) is cache hostile
        // Each k iteration jumps N floats in memory.
        // For any large N this causes cache miss on every access.
        C[i * N + j] += A[i * K + k] * B[k * N + j];
      }
    }
  }
}
