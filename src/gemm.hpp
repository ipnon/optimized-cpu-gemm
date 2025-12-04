#pragma once
#include <cstddef>

void gemm_naive(const float* A, const float* B, float* C, size_t M, size_t N,
                size_t K);
void gemm_blocked(const float* A, const float* B, float* C, size_t M, size_t N,
                  size_t K);
void gemm_simd(const float* A, const float* B, float* C, size_t M, size_t N,
               size_t K);
