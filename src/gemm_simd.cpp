#include <arm_neon.h>

#include <algorithm>

#include "gemm.hpp"

constexpr size_t block_size = 32;

// NEON instructions:
// vld1q_f32 / vst1q_f32 (load/store)
// vdupq_n_f32 (broadcast scalar)
// vfmaq_f32 (fused multiply-add)
// vaddq_f32 (add vectors)

void gemm_simd(const float* A, const float* B, float* C, size_t M, size_t N,
               size_t K) {
  // We start by zeroing C because aligned_alloc in Matrix class returns
  // uninitialized memory.
  for (size_t i = 0; i < M * N; i++) C[i] = 0;

  for (size_t i0 = 0; i0 < M; i0 += block_size) {
    for (size_t k0 = 0; k0 < K; k0 += block_size) {
      for (size_t j0 = 0; j0 < N; j0 += block_size) {
        for (size_t i = i0; i < std::min(i0 + block_size, M); i++) {
          // Step by number of lanes in NEON vector registers
          for (size_t j = j0; j < std::min(j0 + block_size, N); j += 4) {
            // Load current C values before shared dimension loop
            float32x4_t c_vec = vld1q_f32(C + i * N + j);
            for (size_t k = k0; k < std::min(k0 + block_size, K); k++) {
              // Load 4 consecutive B values: B[k, j:j+4]
              // Load needs a pointer to read from memory
              float32x4_t b_vec = vld1q_f32(B + k * N + j);
              // Broadcast one A value: A[i, k] -> all 4 lanes
              // dup with _n takes a scalar number to broadcast
              // The compiler handles register allocation.
              float32x4_t a_broadcast = vdupq_n_f32(A[i * K + k]);
              // Accumulate: C[i, j:j+4] += A_broadcast * B_vec
              c_vec = vfmaq_f32(c_vec, a_broadcast, b_vec);
            }
            // Store result after shared dimension computation
            vst1q_f32(C + i * N + j, c_vec);
          }
        }
      }
    }
  }
}
