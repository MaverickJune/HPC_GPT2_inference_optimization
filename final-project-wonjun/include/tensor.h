#pragma once

#include <vector>

using std::vector;

/* [Tensor Structure] */
struct Tensor {
  size_t ndim = 0;
  size_t shape[5]; // 4->5 to use first dimension as batch
  float *buf = nullptr;

  int cuda_flag = 0;
  float *cuda_buf = nullptr;

  Tensor(const vector<size_t> &shape_);
  Tensor(const vector<size_t> &shape_, float *buf_);
  ~Tensor();

  size_t num_elem();
  size_t num_elem_per_batch();
  void resize_activation(const vector<size_t> &shape_);
};

typedef Tensor Parameter;
typedef Tensor Activation;
