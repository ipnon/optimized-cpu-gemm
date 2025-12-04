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

    double gflops =
        benchmark_gemm(gemm_naive, A.data(), B.data(), C.data(), N, N, N);
    std::cout << N << "x" << N << ": " << gflops << " GFLOPS\n";
  }

  return 0;  // Matrices automatically freed
}
