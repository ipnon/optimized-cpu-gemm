#pragma once
#include <cstddef>
#include <cstdlib>
#include <random>
#include <stdexcept>

class Matrix {
 public:
  Matrix(size_t rows, size_t cols) : rows_(rows), cols_(cols), data_(nullptr) {
    // Aligned allocation for SIMD (64-byte alignment = cache line)
    // Since aligned_alloc requires bytes to be a multiple of alignment (64),
    // round up to a multiple of 64.
    size_t bytes = ((rows * cols * sizeof(float) + 63) / 64) * 64;
    data_ = static_cast<float*>(aligned_alloc(64, bytes));
    if (!data_) throw std::bad_alloc();
  }

  ~Matrix() { std::free(data_); }

  // Delete copy (expensive, prevent accidents)
  Matrix(const Matrix&) = delete;
  Matrix& operator=(const Matrix&) = delete;

  // Allow move (transfer ownership)
  Matrix(Matrix&& other) noexcept  // std::vector will use move instead of copy
                                   // only if move is noexcept
      : rows_(other.rows_), cols_(other.cols_), data_(other.data_) {
    other.data_ = nullptr;
  }

  // Accessors
  float* data() { return data_; }
  const float* data() const { return data_; }
  size_t rows() const { return rows_; }
  size_t cols() const { return cols_; }
  size_t size() const { return rows_ * cols_; }

  float& operator()(size_t i, size_t j) { return data_[i * cols_ + j]; }
  float operator()(size_t i, size_t j) const { return data_[i * cols_ + j]; }

  void fill_random(unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < size(); i++) {
      data_[i] = dist(rng);
    }
  }

  void zero() {
    for (size_t i = 0; i < size(); i++) {
      data_[i] = 0.0f;
    }
  }

 private:
  size_t rows_;
  size_t cols_;
  float* data_;
};
