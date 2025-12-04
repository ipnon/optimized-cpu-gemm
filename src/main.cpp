#include <iostream>

#include "gemm.hpp"

int main() {
  // 2x3 matrix A, 3x2 matrix B -> 2x2 matrix C
  // All 1s: each element of C should equal 3 (the shared dimension K)

  float A[] = {1, 1, 1, 1, 1, 1};
  float B[] = {1, 1, 1, 1, 1, 1};
  float C[4] = {0};  // 2x2 output

  gemm_naive(A, B, C, 2, 2, 3);  // M=2, N=2, K=3

  std::cout << "C = \n";
  std::cout << C[0] << " " << C[1] << "\n";
  std::cout << C[2] << " " << C[3] << "\n";

  return 0;
}
