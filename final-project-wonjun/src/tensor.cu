#include "model.h"
#include <cuda_runtime.h>

/* [Tensor Structure] */
/* Tensor
 * @brief - A multi-dimensional matrix containing elements of a single data
 type.
 * @member - buf  : Data buffer containing elements
 * @member - shape: Shape of tensor from outermost dimension to innermost
 dimension e.g., {{1.0, -0.5, 2.3}, {4.3, 5.6, -7.8}} => shape = {2, 3}
 */

#define CHECK_CUDA(call)                                              \
  do {                                                                \
    cudaError_t status_ = call;                                       \
    if (status_ != cudaSuccess) {                                     \
      fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(status_));                           \
      exit(EXIT_FAILURE);                                             \
    }                                                                 \
  } while (0)

Tensor::Tensor(const vector<size_t> &shape_) {
  ndim = shape_.size();
  for (size_t i = 0; i < ndim; i++) { shape[i] = shape_[i]; }
  size_t N_ = num_elem();
  buf = (float *) calloc(N_, sizeof(float));

  CHECK_CUDA(cudaMalloc(&cuda_buf, N_ * sizeof(float)));
  cuda_flag = 1;
}

Tensor::Tensor(const vector<size_t> &shape_, float *buf_) {
  ndim = shape_.size();
  for (size_t i = 0; i < ndim; i++) { shape[i] = shape_[i]; }
  size_t N_ = num_elem();
  buf = (float *) malloc(N_ * sizeof(float));
  memcpy(buf, buf_, N_ * sizeof(float));

  CHECK_CUDA(cudaMalloc(&cuda_buf, N_ * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(cuda_buf, buf_, N_ * sizeof(float), cudaMemcpyHostToDevice));
  cuda_flag = 1;
}

Tensor::~Tensor() {
  if (buf != nullptr) free(buf);
  if (cuda_flag == 1) cudaFree(cuda_buf);
}

size_t Tensor::num_elem() {
  size_t size = 1;
  for (size_t i = 0; i < ndim; i++) { size *= shape[i]; }
  return size;
}

size_t Tensor::num_elem_per_batch() {
  size_t size = 1;
  for (size_t i = 1; i < ndim; i++) { size *= shape[i]; }
  return size;
}

void Tensor::resize_activation(const vector<size_t> &shape_) {
  ndim = shape_.size();
  for (size_t i = 0; i < ndim; i++) { shape[i] = shape_[i]; }
}