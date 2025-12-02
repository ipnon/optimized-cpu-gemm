#pragma once
#include <cstddef>

void gemm_naive(const float* A, const float* B, float* C, size_t M, size_t N,
                size_t K);
