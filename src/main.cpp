#include <iostream>

#include "benchmark.hpp"
#include "gemm.hpp"
#include "matrix.hpp"

int main() {
  for (size_t N : {256, 512, 1024}) {
    Matrix A(N, N);
    Matrix B(N, N);
    Matrix C(N, N);

    A.fill_random(42);
    B.fill_random(43);

    double gflops_naive =
        benchmark_gemm(gemm_naive, A.data(), B.data(), C.data(), N, N, N);
    double gflops_blocked =
        benchmark_gemm(gemm_blocked, A.data(), B.data(), C.data(), N, N, N);
    double gflops_simd =
        benchmark_gemm(gemm_simd, A.data(), B.data(), C.data(), N, N, N);

    std::cout << N << "x" << N << " naive: " << gflops_naive << " GFLOPS\n";
    std::cout << N << "x" << N << " blocked: " << gflops_blocked << " GFLOPS\n";
    std::cout << N << "x" << N << " simd: " << gflops_simd << " GFLOPS\n";
  }

  return 0;  // Matrices automatically freed
}
