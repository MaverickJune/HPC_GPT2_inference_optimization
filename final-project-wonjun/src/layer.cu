#include "layer.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define NGPU 4
#define MAX_NODE 4
#define ROUND_DIV(x,y) (((x) + (y) - 1) / (y))

// static int mpi_rank, mpi_world_size;

#define CHECK_CUDA(call)                                              \
  do {                                                                \
    cudaError_t status_ = call;                                       \
    if (status_ != cudaSuccess) {                                     \
      fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(status_));                           \
      exit(EXIT_FAILURE);                                             \
    }                                                                 \
  } while (0)

/////////////////////////////////////////////////////////////////////////////////////////////////////

/* Greedy Max Sampling batch
 * @param  [in1]  in: [b, s, V] -> [b, 1, V]
 * @return [ret] out: [b]
 * 's' is the number of tokens in the prompt.
 * 'V' is the number of vocabulary.
 */
vector<int> top1_sampling_batch(Tensor *in) {
  if (in->cuda_buf != NULL) {
    cudaMemcpy(in->buf, in->cuda_buf, in->num_elem() * sizeof(float), cudaMemcpyDeviceToHost);
  }

  size_t s = in->shape[1];
  size_t V = in->shape[2];

  size_t n_batches = in->shape[0];

  vector<int> out(n_batches);
  vector<float> max(n_batches, -INFINITY);

  for (size_t batch = 0; batch < n_batches; batch++) {
    for (size_t i = 0; i < V; i++) {
      if (in->buf[batch * s * V + (s - 1) * V + i] > max[batch]) {
        max[batch] = in->buf[batch * s * V + (s - 1) * V + i];
        out[batch] = i;
      }
    }
  }

  return out;
}

/* GPU version (added) */
// void top_1_sampling_cuda(Tensor *in, Tensor *out) {

// }



/////////////////////////////////////////////////////////////////////////////////////////////////////
// batched operator implementations in CUDA
/////////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////* 1. matmul */////////////////////////////

__global__ void matmul_kernel_batch(float *A, float *B, float *C, int M, int N, int K) {

  // hyperparameters to tune(start)
  const unsigned int BS_M = 16;
  const unsigned int BS_N = 16;
  const unsigned int BS_K = 16;
  const unsigned int pT_M = 4;
  const unsigned int pT_N = 2;
  // hyperparameters to tune(end)

  const unsigned int k_TILED_MAX = (K/BS_K)*BS_K;
  const unsigned int N_TILE_i = M / BS_M;
  const unsigned int N_TILE_j = N / BS_N;
  const unsigned int threadblock_size = (BS_M*BS_N) / (pT_M*pT_N);

  int block_i = blockIdx.y;
  int block_j = blockIdx.x;
  int inblock_i = threadIdx.x / (BS_N/pT_N);
  int inblock_j = threadIdx.x % (BS_N/pT_N);

  int batch_idx = blockIdx.z;

  // save the original pointers for return
  float * orig_A = A + batch_idx * M * K;
  float * orig_B = B + batch_idx * K * N;
  float * orig_C = C + batch_idx * M * N;

  // move the pointers to the correct batch
  A += batch_idx * M * K;
  B += batch_idx * K * N;
  C += batch_idx * M * N;

  // shared memory for A and B
  __shared__ float As[BS_M][BS_K];
  __shared__ float Bs[BS_K][BS_N];

  float per_thread_items[pT_M*pT_N] = {0};
  float temp_reg_A[pT_M] = {0};
  float temp_reg_B[pT_N] = {0};

  if(block_i < N_TILE_i && block_j < N_TILE_j)
  {
    // loop-invariant code motion
    A += (block_i * BS_M) * K;
    B += block_j * BS_N;
    C += (block_i * BS_M) * N + block_j * BS_N;

    const unsigned int As_j = threadIdx.x % BS_K;
    const unsigned int As_i = threadIdx.x / BS_K;
    const unsigned int Bs_j = threadIdx.x % BS_N;
    const unsigned int Bs_i = threadIdx.x / BS_N;
    const unsigned int tb_unit_A = threadblock_size / BS_K;
    const unsigned int tb_unit_B = threadblock_size / BS_N;

    for(int k_tiled=0; k_tiled<k_TILED_MAX; k_tiled+=BS_K){
      // Load A and B into shared memory
      for(int n_unit_row=0; n_unit_row < BS_M; n_unit_row += tb_unit_A){
        As[n_unit_row + As_i][As_j] = A[(n_unit_row + As_i)*K + As_j];
      }
      for(int n_unit_col=0; n_unit_col < BS_K; n_unit_col += tb_unit_B){
        Bs[n_unit_col + Bs_i][Bs_j] = B[(n_unit_col + Bs_i)*N + Bs_j];
      }
      __syncthreads();

      // update A and B position for next iteration
      A += BS_K;
      B += BS_K * N;

      // Compute partial results
      for(int k=0; k<BS_K; ++k){
        // block into registers
        for(int i=0; i<pT_M; ++i){
          temp_reg_A[i] = As[inblock_i*pT_M + i][k];
        }
        for(int j=0; j<pT_N; ++j){
          temp_reg_B[j] = Bs[k][inblock_j*pT_N + j];
        }
        for(int i=0; i<pT_M; ++i){
          for(int j=0; j<pT_N; ++j){
            // per_thread_items[i*pT_N + j] += temp_reg_A[i] * temp_reg_B[j];
            per_thread_items[i*pT_N + j] += temp_reg_A[i] * temp_reg_B[j];
          }
        }
      }
      __syncthreads();
    }

    // perform update for untiled region(K)
    for(int k=k_TILED_MAX; k<K; k++){
      for(int i=0; i<pT_M; i++){
        temp_reg_A[i] = orig_A[(i + pT_M*inblock_i + BS_M*block_i)*K + k];
      }
      for(int j=0; j<pT_N; j++){
        temp_reg_B[j] = orig_B[k*N + (j + pT_N*inblock_j + BS_N*block_j)];
      }
      for(int i=0; i<pT_M; i++){
        for(int j=0; j<pT_N; j++){
          // per_thread_items[i*pT_N + j] += temp_reg_A[i] * temp_reg_B[j];
          per_thread_items[i*pT_N + j] += temp_reg_A[i] * temp_reg_B[j];
        }
      }
    }

    // copy back result to C
    for(int i=0; i<pT_M; i++){
      for(int j=0; j<pT_N; j++){
        C[(i + pT_M*inblock_i)*N + j + pT_N*inblock_j] = per_thread_items[i*pT_N + j];
      }
    }
  }
  else // deal with leftover region
  {
    int global_i = block_i * BS_M + pT_M * inblock_i;
    int global_j = block_j * BS_N + pT_N * inblock_j;

    if(global_i >= M || global_j >= N) return; //check boundary condition

    for(int i=0; i<pT_M; i++){
      int global_i_iter = global_i + i;
      if(global_i_iter >= M) break; 

      for(int j=0; j<pT_N; j++){
        int global_j_iter = global_j + j;
        if(global_j_iter >= N) break;

        float sum = 0.0;
        for(int k=0; k<K; k++){
          sum += orig_A[global_i_iter*K + k] * orig_B[k*N + global_j_iter];
        }

        orig_C[global_i_iter*N + global_j_iter] = sum;
      }
    }
  }
}


__global__ void matmul_kernel_batch_iter(float *A, float *B, float *C, int M, int N, int K) {

  // hyperparameters to tune(start)
  const unsigned int BS_M = 1;
  const unsigned int BS_N = 16;
  const unsigned int BS_K = 16;
  const unsigned int pT_M = 1;
  const unsigned int pT_N = 1;
  // hyperparameters to tune(end)

  const unsigned int k_TILED_MAX = (K/BS_K)*BS_K;
  const unsigned int N_TILE_i = M / BS_M;
  const unsigned int N_TILE_j = N / BS_N;
  const unsigned int threadblock_size = (BS_M*BS_N) / (pT_M*pT_N);

  int block_i = blockIdx.y;
  int block_j = blockIdx.x;
  int inblock_i = threadIdx.x / (BS_N/pT_N);
  int inblock_j = threadIdx.x % (BS_N/pT_N);

  int batch_idx = blockIdx.z;

  // save the original pointers for return
  float * orig_A = A + batch_idx * M * K;
  float * orig_B = B + batch_idx * K * N;
  float * orig_C = C + batch_idx * M * N;

  // move the pointers to the correct batch
  A += batch_idx * M * K;
  B += batch_idx * K * N;
  C += batch_idx * M * N;

  // shared memory for A and B
  __shared__ float As[BS_M][BS_K];
  __shared__ float Bs[BS_K][BS_N];

  float per_thread_items[pT_M*pT_N] = {0};
  float temp_reg_A[pT_M] = {0};
  float temp_reg_B[pT_N] = {0};

  if(block_i < N_TILE_i && block_j < N_TILE_j)
  {
    // loop-invariant code motion
    A += (block_i * BS_M) * K;
    B += block_j * BS_N;
    C += (block_i * BS_M) * N + block_j * BS_N;

    const unsigned int As_j = threadIdx.x % BS_K;
    const unsigned int As_i = threadIdx.x / BS_K;
    const unsigned int Bs_j = threadIdx.x % BS_N;
    const unsigned int Bs_i = threadIdx.x / BS_N;
    const unsigned int tb_unit_A = threadblock_size / BS_K;
    const unsigned int tb_unit_B = threadblock_size / BS_N;

    for(int k_tiled=0; k_tiled<k_TILED_MAX; k_tiled+=BS_K){
      // Load A and B into shared memory
      for(int n_unit_row=0; n_unit_row < BS_M; n_unit_row += tb_unit_A){
        As[n_unit_row + As_i][As_j] = A[(n_unit_row + As_i)*K + As_j];
      }
      for(int n_unit_col=0; n_unit_col < BS_K; n_unit_col += tb_unit_B){
        Bs[n_unit_col + Bs_i][Bs_j] = B[(n_unit_col + Bs_i)*N + Bs_j];
      }
      __syncthreads();

      // update A and B position for next iteration
      A += BS_K;
      B += BS_K * N;

      // Compute partial results
      for(int k=0; k<BS_K; ++k){
        // block into registers
        for(int i=0; i<pT_M; ++i){
          temp_reg_A[i] = As[inblock_i*pT_M + i][k];
        }
        for(int j=0; j<pT_N; ++j){
          temp_reg_B[j] = Bs[k][inblock_j*pT_N + j];
        }
        for(int i=0; i<pT_M; ++i){
          for(int j=0; j<pT_N; ++j){
            // per_thread_items[i*pT_N + j] += temp_reg_A[i] * temp_reg_B[j];
            per_thread_items[i*pT_N + j] += temp_reg_A[i] * temp_reg_B[j];
          }
        }
      }
      __syncthreads();
    }

    // perform update for untiled region(K)
    for(int k=k_TILED_MAX; k<K; k++){
      for(int i=0; i<pT_M; i++){
        temp_reg_A[i] = orig_A[(i + pT_M*inblock_i + BS_M*block_i)*K + k];
      }
      for(int j=0; j<pT_N; j++){
        temp_reg_B[j] = orig_B[k*N + (j + pT_N*inblock_j + BS_N*block_j)];
      }
      for(int i=0; i<pT_M; i++){
        for(int j=0; j<pT_N; j++){
          // per_thread_items[i*pT_N + j] += temp_reg_A[i] * temp_reg_B[j];
          per_thread_items[i*pT_N + j] += temp_reg_A[i] * temp_reg_B[j];
        }
      }
    }

    // copy back result to C
    for(int i=0; i<pT_M; i++){
      for(int j=0; j<pT_N; j++){
        C[(i + pT_M*inblock_i)*N + j + pT_N*inblock_j] = per_thread_items[i*pT_N + j];
      }
    }
  }
  else // deal with leftover region
  {
    int global_i = block_i * BS_M + pT_M * inblock_i;
    int global_j = block_j * BS_N + pT_N * inblock_j;

    if(global_i >= M || global_j >= N) return; //check boundary condition

    for(int i=0; i<pT_M; i++){
      int global_i_iter = global_i + i;
      if(global_i_iter >= M) break; 

      for(int j=0; j<pT_N; j++){
        int global_j_iter = global_j + j;
        if(global_j_iter >= N) break;

        float sum = 0.0;
        for(int k=0; k<K; k++){
          sum += orig_A[global_i_iter*K + k] * orig_B[k*N + global_j_iter];
        }

        orig_C[global_i_iter*N + global_j_iter] = sum;
      }
    }
  }
}

/* Matmul for batching + cuda
 * @param [in1]  in1: [b, M, K]
 * @param [in2]  in2: [b, K, N]
 * @param [out]  out: [b, M, N]
 */

void matmul_batch_cuda(Tensor *in1, Tensor *in2, Tensor *out, size_t token_num) {

  if (token_num == 0) {
    // hyperparameters to tune(start)
    const unsigned int BS_M = 16;
    const unsigned int BS_N = 16;
    // const unsigned int BS_K = 16;
    const unsigned int pT_M = 4;
    const unsigned int pT_N = 2;
    // hyperparameters to tune(end)

    size_t M = in1->shape[1];
    size_t K = in1->shape[2];
    size_t N = in2->shape[2];

    size_t n_batches = in1->shape[0];

    // kernel launch
    dim3 blockDim((BS_N * BS_M)/(pT_N * pT_M));
    dim3 gridDim(ROUND_DIV(N, BS_N), ROUND_DIV(M, BS_M), n_batches);
    matmul_kernel_batch<<<gridDim, blockDim>>>(in1->cuda_buf, in2->cuda_buf, out->cuda_buf, M, N, K);

    CHECK_CUDA(cudaGetLastError());
  }
  else {
    // hyperparameters to tune(start)
    const unsigned int BS_M = 1;
    const unsigned int BS_N = 16;
    // const unsigned int BS_K = 16;
    const unsigned int pT_M = 1;
    const unsigned int pT_N = 1;
    // hyperparameters to tune(end)

    size_t M = in1->shape[1];
    size_t K = in1->shape[2];
    size_t N = in2->shape[2];

    size_t n_batches = in1->shape[0];

    // kernel launch
    dim3 blockDim((BS_N * BS_M)/(pT_N * pT_M));
    dim3 gridDim(ROUND_DIV(N, BS_N), ROUND_DIV(M, BS_M), n_batches);
    matmul_kernel_batch_iter<<<gridDim, blockDim>>>(in1->cuda_buf, in2->cuda_buf, out->cuda_buf, M, N, K);

    CHECK_CUDA(cudaGetLastError());
  }
}



/////////////////////////////* 2. linear */////////////////////////////
__global__ void linear_kernel_batch(float *A, float *B, float *C, float *bias, int M, int N, int K, size_t token_num) {

  // hyperparameters to tune(start), keep this value same as the one in launch code(below)
  const unsigned int BS_M = 64;
  const unsigned int BS_N = 64;
  const unsigned int BS_K = 32;
  const unsigned int pT_M = 8;
  const unsigned int pT_N = 4;
  // hyperparameters to tune(end)

  const unsigned int k_TILED_MAX = (K/BS_K)*BS_K;
  const unsigned int N_TILE_i = M / BS_M;
  const unsigned int N_TILE_j = N / BS_N;
  const unsigned int threadblock_size = (BS_M*BS_N) / (pT_M*pT_N);

  int block_i = blockIdx.y;
  int block_j = blockIdx.x;
  int inblock_i = threadIdx.x / (BS_N/pT_N);
  int inblock_j = threadIdx.x % (BS_N/pT_N);

  // int batch_idx = blockIdx.z; // modified

  // save the original pointers for return
  // float * orig_A = A + batch_idx * M * K; // modified
  // float * orig_B = B;
  // float * orig_C = C + batch_idx * M * N;
  float * orig_A = A;
  float * orig_B = B;
  float * orig_C = C;

  // move the pointers to the correct batch
  // A += batch_idx * M * K; // modified
  // C += batch_idx * M * N;

  // shared memory for A and B
  __shared__ float As[BS_M][BS_K];
  __shared__ float Bs[BS_K][BS_N];
  __shared__ float bias_s[BS_N];

  float per_thread_items[pT_M*pT_N] = {0};
  float temp_reg_A[pT_M] = {0};
  float temp_reg_B[pT_N] = {0};

  if(block_i < N_TILE_i && block_j < N_TILE_j)
  {
    // loop-invariant code motion
    A += (block_i * BS_M) * K;
    B += block_j * BS_N;
    C += (block_i * BS_M) * N + block_j * BS_N;

    const unsigned int As_j = threadIdx.x % BS_K;
    const unsigned int As_i = threadIdx.x / BS_K;
    const unsigned int Bs_j = threadIdx.x % BS_N;
    const unsigned int Bs_i = threadIdx.x / BS_N;
    const unsigned int tb_unit_A = threadblock_size / BS_K;
    const unsigned int tb_unit_B = threadblock_size / BS_N;

    for(int k_tiled=0; k_tiled<k_TILED_MAX; k_tiled+=BS_K){
      // Load A and B into shared memory
      for(int n_unit_row=0; n_unit_row < BS_M; n_unit_row += tb_unit_A){
        As[n_unit_row + As_i][As_j] = A[(n_unit_row + As_i)*K + As_j];
      }
      for(int n_unit_col=0; n_unit_col < BS_K; n_unit_col += tb_unit_B){
        Bs[n_unit_col + Bs_i][Bs_j] = B[(n_unit_col + Bs_i)*N + Bs_j];
      }
      __syncthreads();

      // update A and B position for next iteration
      A += BS_K;
      B += BS_K * N;

      // Compute partial results
      for(int k=0; k<BS_K; ++k){
        // block into registers
        for(int i=0; i<pT_M; ++i){
          temp_reg_A[i] = As[inblock_i*pT_M + i][k];
        }
        for(int j=0; j<pT_N; ++j){
          temp_reg_B[j] = Bs[k][inblock_j*pT_N + j];
        }
        for(int i=0; i<pT_M; ++i){
          for(int j=0; j<pT_N; ++j){
            // per_thread_items[i*pT_N + j] += temp_reg_A[i] * temp_reg_B[j];
            per_thread_items[i*pT_N + j] += temp_reg_A[i] * temp_reg_B[j];
          }
        }
      }
      __syncthreads();
    }

    // perform update for untiled region(K)
    for(int k=k_TILED_MAX; k<K; k++){
      for(int i=0; i<pT_M; i++){
        temp_reg_A[i] = orig_A[(i + pT_M*inblock_i + BS_M*block_i)*K + k];
      }
      for(int j=0; j<pT_N; j++){
        temp_reg_B[j] = orig_B[k*N + (j + pT_N*inblock_j + BS_N*block_j)];
      }
      for(int i=0; i<pT_M; i++){
        for(int j=0; j<pT_N; j++){
          // per_thread_items[i*pT_N + j] += temp_reg_A[i] * temp_reg_B[j];
          per_thread_items[i*pT_N + j] += temp_reg_A[i] * temp_reg_B[j];
        }
      }
    }

    if(inblock_i == 0){
      for(int j=0; j<pT_N; j++){
        bias_s[j + pT_N*inblock_j] = bias[j + pT_N*inblock_j + BS_N*block_j];
      }
    }
    __syncthreads();

    // copy back result to C
    for(int i=0; i<pT_M; i++){
      for(int j=0; j<pT_N; j++){
        // C[(i + pT_M*inblock_i)*N + j + pT_N*inblock_j] = per_thread_items[i*pT_N + j] + bias[j + pT_N*inblock_j + BS_N*block_j];
        C[(i + pT_M*inblock_i)*N + j + pT_N*inblock_j] = per_thread_items[i*pT_N + j] + bias_s[j + pT_N*inblock_j];
      }
    }
  }
  else // deal with leftover region
  {
    int global_i = block_i * BS_M + pT_M * inblock_i;
    int global_j = block_j * BS_N + pT_N * inblock_j;

    if(global_i >= M || global_j >= N) return; //check boundary condition

    for(int i=0; i<pT_M; i++){
      int global_i_iter = global_i + i;
      if(global_i_iter >= M) break; 

      for(int j=0; j<pT_N; j++){
        int global_j_iter = global_j + j;
        if(global_j_iter >= N) break;

        float sum = 0.0;
        for(int k=0; k<K; k++){
          sum += orig_A[global_i_iter*K + k] * orig_B[k*N + global_j_iter];
        }

        orig_C[global_i_iter*N + global_j_iter] = sum + bias[global_j_iter];
      }
    }
  }
}

__global__ void linear_kernel_batch_iter(float *A, float *B, float *C, float *bias, int M, int N, int K, size_t token_num) {

  // hyperparameters to tune(start), keep this value same as the one in launch code(below)
  const unsigned int BS_M = 64;
  const unsigned int BS_N = 64;
  const unsigned int BS_K = 32;
  const unsigned int pT_M = 8;
  const unsigned int pT_N = 4;
  // hyperparameters to tune(end)

  const unsigned int k_TILED_MAX = (K/BS_K)*BS_K;
  const unsigned int N_TILE_i = M / BS_M;
  const unsigned int N_TILE_j = N / BS_N;
  const unsigned int threadblock_size = (BS_M*BS_N) / (pT_M*pT_N);

  int block_i = blockIdx.y;
  int block_j = blockIdx.x;
  int inblock_i = threadIdx.x / (BS_N/pT_N);
  int inblock_j = threadIdx.x % (BS_N/pT_N);

  // int batch_idx = blockIdx.z; // modified

  // save the original pointers for return
  // float * orig_A = A + batch_idx * M * K; // modified
  // float * orig_B = B;
  // float * orig_C = C + batch_idx * M * N;
  float * orig_A = A;
  float * orig_B = B;
  float * orig_C = C;

  // move the pointers to the correct batch
  // A += batch_idx * M * K; // modified
  // C += batch_idx * M * N;

  // shared memory for A and B
  __shared__ float As[BS_M][BS_K];
  __shared__ float Bs[BS_K][BS_N];
  __shared__ float bias_s[BS_N];

  float per_thread_items[pT_M*pT_N] = {0};
  float temp_reg_A[pT_M] = {0};
  float temp_reg_B[pT_N] = {0};

  if(block_i < N_TILE_i && block_j < N_TILE_j)
  {
    // loop-invariant code motion
    A += (block_i * BS_M) * K;
    B += block_j * BS_N;
    C += (block_i * BS_M) * N + block_j * BS_N;

    const unsigned int As_j = threadIdx.x % BS_K;
    const unsigned int As_i = threadIdx.x / BS_K;
    const unsigned int Bs_j = threadIdx.x % BS_N;
    const unsigned int Bs_i = threadIdx.x / BS_N;
    const unsigned int tb_unit_A = threadblock_size / BS_K;
    const unsigned int tb_unit_B = threadblock_size / BS_N;

    for(int k_tiled=0; k_tiled<k_TILED_MAX; k_tiled+=BS_K){
      // Load A and B into shared memory
      for(int n_unit_row=0; n_unit_row < BS_M; n_unit_row += tb_unit_A){
        As[n_unit_row + As_i][As_j] = A[(n_unit_row + As_i)*K + As_j];
      }
      for(int n_unit_col=0; n_unit_col < BS_K; n_unit_col += tb_unit_B){
        Bs[n_unit_col + Bs_i][Bs_j] = B[(n_unit_col + Bs_i)*N + Bs_j];
      }
      __syncthreads();

      // update A and B position for next iteration
      A += BS_K;
      B += BS_K * N;

      // Compute partial results
      for(int k=0; k<BS_K; ++k){
        // block into registers
        for(int i=0; i<pT_M; ++i){
          temp_reg_A[i] = As[inblock_i*pT_M + i][k];
        }
        for(int j=0; j<pT_N; ++j){
          temp_reg_B[j] = Bs[k][inblock_j*pT_N + j];
        }
        for(int i=0; i<pT_M; ++i){
          for(int j=0; j<pT_N; ++j){
            // per_thread_items[i*pT_N + j] += temp_reg_A[i] * temp_reg_B[j];
            per_thread_items[i*pT_N + j] += temp_reg_A[i] * temp_reg_B[j];
          }
        }
      }
      __syncthreads();
    }

    // perform update for untiled region(K)
    for(int k=k_TILED_MAX; k<K; k++){
      for(int i=0; i<pT_M; i++){
        temp_reg_A[i] = orig_A[(i + pT_M*inblock_i + BS_M*block_i)*K + k];
      }
      for(int j=0; j<pT_N; j++){
        temp_reg_B[j] = orig_B[k*N + (j + pT_N*inblock_j + BS_N*block_j)];
      }
      for(int i=0; i<pT_M; i++){
        for(int j=0; j<pT_N; j++){
          // per_thread_items[i*pT_N + j] += temp_reg_A[i] * temp_reg_B[j];
          per_thread_items[i*pT_N + j] += temp_reg_A[i] * temp_reg_B[j];
        }
      }
    }

    if(inblock_i == 0){
      for(int j=0; j<pT_N; j++){
        bias_s[j + pT_N*inblock_j] = bias[j + pT_N*inblock_j + BS_N*block_j];
      }
    }
    __syncthreads();

    // copy back result to C
    for(int i=0; i<pT_M; i++){
      for(int j=0; j<pT_N; j++){
        // C[(i + pT_M*inblock_i)*N + j + pT_N*inblock_j] = per_thread_items[i*pT_N + j] + bias[j + pT_N*inblock_j + BS_N*block_j];
        C[(i + pT_M*inblock_i)*N + j + pT_N*inblock_j] = per_thread_items[i*pT_N + j] + bias_s[j + pT_N*inblock_j];
      }
    }
  }
  else // deal with leftover region
  {
    int global_i = block_i * BS_M + pT_M * inblock_i;
    int global_j = block_j * BS_N + pT_N * inblock_j;

    if(global_i >= M || global_j >= N) return; //check boundary condition

    for(int i=0; i<pT_M; i++){
      int global_i_iter = global_i + i;
      if(global_i_iter >= M) break; 

      for(int j=0; j<pT_N; j++){
        int global_j_iter = global_j + j;
        if(global_j_iter >= N) break;

        float sum = 0.0;
        for(int k=0; k<K; k++){
          sum += orig_A[global_i_iter*K + k] * orig_B[k*N + global_j_iter];
        }

        orig_C[global_i_iter*N + global_j_iter] = sum + bias[global_j_iter];
      }
    }
  }
}


/* Linear for batch + cuda
 * @param [in1]  in: [b, M, K]
 * @param [in2]   w: [K, N]
 * @param [in3]   b: [N]
 * @param [out] out: [b, M, N]
 */

void linear_batch_cuda(Tensor *in, Tensor *w, Tensor *b, Tensor *out, size_t token_num) {

  if (token_num == 0) {
    // hyperparameters to tune(start)
    const unsigned int BS_M = 64;
    const unsigned int BS_N = 64;
    // const unsigned int BS_K = 32;
    const unsigned int pT_M = 8;
    const unsigned int pT_N = 4;
    // hyperparameters to tune(end)

    size_t n_batches = in->shape[0];
    size_t M = in->shape[1];
    size_t K = in->shape[2];
    size_t N = w->shape[1];

    // kernel launch
    dim3 blockDim((BS_N * BS_M)/(pT_N * pT_M));
    dim3 gridDim(ROUND_DIV(N, BS_N), ROUND_DIV(M * n_batches, BS_M));
    linear_kernel_batch<<<gridDim, blockDim>>>(in->cuda_buf, w->cuda_buf, out->cuda_buf, b->cuda_buf, M * n_batches, N, K, token_num);

    CHECK_CUDA(cudaGetLastError());
  }
  else {
    // hyperparameters to tune(start)
    const unsigned int BS_M = 64;
    const unsigned int BS_N = 64;
    // const unsigned int BS_K = 32;
    const unsigned int pT_M = 8;
    const unsigned int pT_N = 4;
    // hyperparameters to tune(end)

    size_t n_batches = in->shape[0];
    size_t M = in->shape[1];
    size_t K = in->shape[2];
    size_t N = w->shape[1];

    // kernel launch
    dim3 blockDim((BS_N * BS_M)/(pT_N * pT_M));
    dim3 gridDim(ROUND_DIV(N, BS_N), ROUND_DIV(M * n_batches, BS_M));
    linear_kernel_batch_iter<<<gridDim, blockDim>>>(in->cuda_buf, w->cuda_buf, out->cuda_buf, b->cuda_buf, M * n_batches, N, K, token_num);

    CHECK_CUDA(cudaGetLastError());
  }
}



/////////////////////////////* 3. token_pos_embedding */////////////////////////////
__global__ void token_pos_embedding_kernel_batch(int s, int H, int *in, float *wte, float *wpe, float *out) {
  size_t global_idx_per_batch = blockIdx.x * blockDim.x + threadIdx.x;
  size_t i = global_idx_per_batch / H;
  size_t j = global_idx_per_batch % H;

  size_t batch_idx = blockIdx.y;

  in += batch_idx * (s);
  out += batch_idx * (s * H);

  if (global_idx_per_batch < s * H) {
    out[i * H + j] = wte[in[i] * H + j] + wpe[i * H + j];
  }
}

__global__ void token_pos_embedding_kernel_batch_iter(int H, int token_num, int *in, float *wte, float *wpe, float *out) {
  size_t global_idx_per_batch = blockIdx.x * blockDim.x + threadIdx.x;
  size_t batch_idx = blockIdx.y;

  in += batch_idx * (1);
  out += batch_idx * (1 * H);

  if (global_idx_per_batch < 1 * H) {
    out[global_idx_per_batch] = wte[in[0] * H + global_idx_per_batch] + wpe[(15 + token_num) * H + global_idx_per_batch];
  }

}

/* Token Pos Embedding for batch + cuda
 * @param [in1]  in: [b, s]
 * @param [in2] wte: [V, H]
 * @param [in3] wpe: [s, H]
 * @param [out] out: [b, s, H]
 * 
 * if the token_num >= 1,  --> s = 1 
 */

void token_pos_embedding_batch_cuda(vector<int> in[], Tensor *wte, Tensor *wpe, Tensor *out, size_t token_num) {
  size_t n_batches = out->shape[0];
  int *d_in;

  if (token_num == 0) {
    // this is the prompt phase
    size_t s = in[0].size();
    size_t H = wte->shape[1];

    cudaMalloc(&d_in, n_batches * s * sizeof(int));
    for (size_t i = 0; i < n_batches; i++) {
      CHECK_CUDA(cudaMemcpy(d_in + i * s, in[i].data(), s * sizeof(int), cudaMemcpyHostToDevice));
    }

    const unsigned int BLOCK_SIZE = 256;
    dim3 gridDim(ROUND_DIV(s * H, BLOCK_SIZE), n_batches);
    dim3 blockDim(BLOCK_SIZE);

    token_pos_embedding_kernel_batch<<<gridDim, blockDim>>>(s, H, d_in, wte->cuda_buf, wpe->cuda_buf, out->cuda_buf);
  }
  else {
    // this is auto-regression phase
    size_t s = 1;
    size_t H = wte->shape[1];

    cudaMalloc(&d_in, n_batches * s * sizeof(int));
    for (size_t i = 0; i < n_batches; i++) {
      CHECK_CUDA(cudaMemcpy(d_in + i * s, &in[i][15 + token_num], s * sizeof(int), cudaMemcpyHostToDevice));
    }

    const unsigned int BLOCK_SIZE = 256;
    dim3 gridDim(ROUND_DIV(s * H, BLOCK_SIZE), n_batches);

    token_pos_embedding_kernel_batch_iter<<<gridDim, BLOCK_SIZE>>>(H, token_num, d_in, wte->cuda_buf, wpe->cuda_buf, out->cuda_buf);
  }

  // free all the allocated memory
  cudaFree(d_in);
}



/////////////////////////////* 4. gelu */////////////////////////////
__global__ void gelu_kernel_batch(int N, float *inout) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    float x = inout[idx];
    inout[idx] = 0.5 * x * (1.f + tanh(sqrt(2.f / MATH_PI) * (x + 0.044715f * x * x * x)));
  }
}

void gelu_batch_cuda(Tensor *inout) {
  size_t N = inout->num_elem();

  // define grid and block size
  const unsigned int BLOCK_SIZE = 256;
  dim3 gridDim(ROUND_DIV(N, BLOCK_SIZE));
  dim3 blockDim(BLOCK_SIZE);

  // call kernel
  gelu_kernel_batch<<<gridDim, blockDim>>>(N, inout->cuda_buf);
}



/////////////////////////////* 5. layernorm */////////////////////////////
__global__ void layer_norm_batch(float *inout, float *gamma, float *beta, int s, int H) {
  size_t global_idx_per_batch = blockIdx.x * blockDim.x + threadIdx.x;
  size_t batch_idx = blockIdx.y;

  float eps = 1e-5;

  inout += batch_idx * (s * H);

  if (global_idx_per_batch < s) {
    float mean = 0;
    float var = 0;

    for (size_t i = 0; i < H; i++) {
      mean += inout[global_idx_per_batch * H + i];
      var += inout[global_idx_per_batch * H + i] * inout[global_idx_per_batch * H + i];
    }

    mean /= H;
    var = var / H - mean * mean;

    for (int j = 0; j < H; j++) {
      inout[global_idx_per_batch * H + j] = (inout[global_idx_per_batch * H + j] - mean) * (1.0 / sqrt(var + eps)) * gamma[j] + beta[j];
    }
  }
}

const unsigned int BLOCK_SIZE_LAYER_NORM = 768;
__global__ void layer_norm_batch_reduction(float *inout, float *gamma, float *beta, int s, int H) {
  __shared__ float sum_board[BLOCK_SIZE_LAYER_NORM];
  __shared__ float square_sum_board[BLOCK_SIZE_LAYER_NORM];

  __shared__ float mean;
  __shared__ float var;

  float eps = 1e-5;

  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int idx = bid * H + tid;

  float value = (tid < H) ? inout[idx] : 0;
  sum_board[tid] = value;
  square_sum_board[tid] = value * value;
  if (tid == 0) {
    mean = 0.0f;
    var = 0.0f;
  }
  __syncthreads();

  float sum_pt = (tid < H) ? sum_board[tid] : 0;
  float square_sum_pt = (tid < H) ? square_sum_board[tid] : 0;
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    sum_pt += __shfl_down_sync(0xffffffff, sum_pt, offset);
    square_sum_pt += __shfl_down_sync(0xffffffff, square_sum_pt, offset);
  }

  if (tid % warpSize == 0) {
    atomicAdd(&mean, sum_pt);
    atomicAdd(&var, square_sum_pt);
  }
  __syncthreads();

  if (tid == 0) {
    mean /= H;
    var = var / H - mean * mean;
  }
  __syncthreads();

  if (tid < H) {
    inout[idx] = (inout[idx] - mean) * (1.0 / sqrt(var + eps)) * gamma[tid] + beta[tid];
  }
}

/* Layer Normalization batch + cuda
 * @param [in1 & out] inout: [b, s, H]
 * @param [in2]       gamma: [H]
 * @param [in3]        beta: [H]
 * 's' is the number of tokens in the prompt.
 * 'H' is the hidden dimension.
 */
void layer_norm_batch_cuda(Tensor *inout, Tensor *gamma, Tensor *beta) {
  size_t s = inout->shape[1];
  size_t H = inout->shape[2];

  size_t n_batches = inout->shape[0];

  assert(H <= BLOCK_SIZE_LAYER_NORM);

  // // define grid and block size
  // const unsigned int BLOCK_SIZE = 256; // 256
  // // dim3 gridDim(ROUND_DIV(s * n_batches, BLOCK_SIZE));
  // dim3 gridDim(ROUND_DIV(s, BLOCK_SIZE), n_batches); // modified
  // dim3 blockDim(BLOCK_SIZE);

  // // call kernel
  // layer_norm_batch<<<gridDim, blockDim>>>(inout->cuda_buf, gamma->cuda_buf, beta->cuda_buf, s, H);

  // define grid and block size
  dim3 gridDim(n_batches * s * ROUND_DIV(H, BLOCK_SIZE_LAYER_NORM));
  dim3 blockDim(BLOCK_SIZE_LAYER_NORM);

  // call kernel
  layer_norm_batch_reduction<<<gridDim, blockDim>>>(inout->cuda_buf, gamma->cuda_buf, beta->cuda_buf, s, H);
}



/////////////////////////////* 6. scaling */////////////////////////////
__global__ void scaling_kernel_batch(float *inout, float scale, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    inout[idx] *= scale;
  }
}

void scaling_batch_cuda(Tensor *inout, float scale) {
  size_t N = inout->num_elem();

  // define grid and block size
  const unsigned int BLOCK_SIZE = 256;
  dim3 gridDim(ROUND_DIV(N, BLOCK_SIZE));
  dim3 blockDim(BLOCK_SIZE);

  // call kernel
  scaling_kernel_batch<<<gridDim, blockDim>>>(inout->cuda_buf, scale, N);
}



/////////////////////////////* 7. softmax */////////////////////////////
// from : https://stackoverflow.com/questions/17399119/how-do-i-use-atomicmax-on-floating-point-values-in-cuda
__device__ __forceinline__ float atomicMaxFloat (float * addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) :
         __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));

    return old;
}

const int BLOCK_SIZE_SOFTMAX = 64;
__global__ void softmax_kernel_batch(float* inout, int H) {
  __shared__ float shared_data[BLOCK_SIZE_SOFTMAX];
  __shared__ float max_board[BLOCK_SIZE_SOFTMAX];
  __shared__ float max_val_shared; // added
  __shared__ float sum_exp_shared;

  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int idx = bid * H + tid;

  // Load input data into shared memory
  float value = (tid < H) ? inout[idx] : -INFINITY;
  shared_data[tid] = value;
  max_board[tid] = value;
  // if (tid == 0) sum_exp_shared = 0.0f;
  if (tid == 0) { // added
    sum_exp_shared = 0.0f;
    max_val_shared = -INFINITY;
  }
  __syncthreads();

  // Compute max value using block reduction
  for (int stride = BLOCK_SIZE_SOFTMAX / 2; stride > 0; stride /= 2) {
      if (tid < stride && tid + stride < H) {
          max_board[tid] = max(max_board[tid], max_board[tid + stride]);
      }
      __syncthreads();
  }

  if (tid == 0) {
      max_val_shared = max_board[0];
  }
  
  __syncthreads();

  // Step 2: Compute exponentials and their sum
  if (tid < H) {
    shared_data[tid] = expf(shared_data[tid] - max_val_shared);
  }
  __syncthreads();

  // Compute sum of exponentials using block reduction
  float sum_exp = (tid < H) ? shared_data[tid] : 0.0f;
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);
  }

  if ((tid % (warpSize)) == 0) { // tid & (warpSize - 1) == 0
    atomicAdd(&sum_exp_shared, sum_exp);
  }
  __syncthreads();

  // Step 3: Compute softmax output
  if (tid < H) {
    inout[idx] = shared_data[tid] / sum_exp_shared;
  }
}

/* Softmax batch + cuda (w/ Max Trick)
 * @param [in & out] inout: [b, s, H]
 * 's' is the number of tokens in the prompt.
 * 'H' is the hidden dimension.
 */
void softmax_batch_cuda(Tensor *inout) {

  /*
  assumption
  tokens per prompt = 16
  (BLOCK_SIZE_SOFTMAX = 64) should be bigger than the row of the inout matrix
  currently, it is 16
  */

  size_t n = inout->num_elem();
  size_t s = inout->shape[1];
  size_t H = inout->shape[2];

  size_t n_batches = inout->shape[0];

  // define grid and block size
  dim3 gridDim(n_batches * s * ROUND_DIV(H, BLOCK_SIZE_SOFTMAX));
  dim3 blockDim(BLOCK_SIZE_SOFTMAX);

  // call kernel
  softmax_kernel_batch<<<gridDim, blockDim>>>(inout->cuda_buf, H);
}



/////////////////////////////* 8. transpose (cuda, non-batched version) */////////////////////////////
__global__ void transpose_kernel(float *in, float *out, int M, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < M * N) {
    int i = idx / N;
    int j = idx % N;
    out[j * M + i] = in[i * N + j];
  }
}

/* Transpose cuda
 * @param [in1]  in: [M, N]
 * @param [out] out: [N, M]
 */
void transpose_cuda(Tensor *in, Tensor *out) {
  size_t M = in->shape[0];
  size_t N = in->shape[1];

  // define grid and block size
  const unsigned int BLOCK_SIZE = 256;
  dim3 gridDim(ROUND_DIV(M * N, BLOCK_SIZE));
  dim3 blockDim(BLOCK_SIZE);

  // call kernel
  transpose_kernel<<<gridDim, blockDim>>>(in->cuda_buf, out->cuda_buf, M, N);
}



/////////////////////////////* 9. transpose (cuda, batched version) *///////////////////////////// 
__global__ void transpose_kernel_batch(float *in, float *out, int M, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int batch_idx = blockIdx.y;

  in += batch_idx * (M * N);
  out += batch_idx * (N * M);

  if (idx < M * N) {
    int i = idx / N;
    int j = idx % N;
    out[j * M + i] = in[i * N + j];
  }
}

/* Transpose_batch + cuda
 * @param [in1]  in: [b, M, N]
 * @param [out] out: [b, N, M]
 */
void transpose_batch_cuda(Tensor *in, Tensor *out) {
  size_t M = in->shape[1];
  size_t N = in->shape[2];

  size_t n_batches = in->shape[0];

  // define grid and block size
  const unsigned int BLOCK_SIZE = 256;
  dim3 gridDim(ROUND_DIV(M * N, BLOCK_SIZE), n_batches);
  dim3 blockDim(BLOCK_SIZE);

  // call kernel
  transpose_kernel_batch<<<gridDim, blockDim>>>(in->cuda_buf, out->cuda_buf, M, N);
}



/////////////////////////////* 10. matmul_batch_final *///////////////////////////// 

// this is used to calculate the final output of the model
__global__ void matmul_kernel(float *A, float *B, float *C, int M, int N, int K, size_t token_num) {

  // hyperparameters to tune(start), keep this value same as the one in launch code(below)
  const unsigned int BS_M = 32;
  const unsigned int BS_N = 64;
  const unsigned int BS_K = 16;
  const unsigned int pT_M = 8;
  const unsigned int pT_N = 4;
  // hyperparameters to tune(end)

  const unsigned int k_TILED_MAX = (K/BS_K)*BS_K;
  const unsigned int N_TILE_i = M / BS_M;
  const unsigned int N_TILE_j = N / BS_N;
  const unsigned int threadblock_size = (BS_M*BS_N) / (pT_M*pT_N);

  int block_i = blockIdx.y;
  int block_j = blockIdx.x;
  int inblock_i = threadIdx.x / (BS_N/pT_N);
  int inblock_j = threadIdx.x % (BS_N/pT_N);

  // save the original pointers for return
  float * orig_A = A;
  float * orig_B = B;
  float * orig_C = C;

  // shared memory for A and B
  __shared__ float As[BS_M][BS_K];
  __shared__ float Bs[BS_K][BS_N];

  float per_thread_items[pT_M*pT_N] = {0};
  float temp_reg_A[pT_M] = {0};
  float temp_reg_B[pT_N] = {0};

  if(block_i < N_TILE_i && block_j < N_TILE_j)
  {
    // loop-invariant code motion
    A += (block_i * BS_M) * K;
    B += block_j * BS_N;
    C += (block_i * BS_M) * N + block_j * BS_N;

    const unsigned int As_j = threadIdx.x % BS_K;
    const unsigned int As_i = threadIdx.x / BS_K;
    const unsigned int Bs_j = threadIdx.x % BS_N;
    const unsigned int Bs_i = threadIdx.x / BS_N;
    const unsigned int tb_unit_A = threadblock_size / BS_K;
    const unsigned int tb_unit_B = threadblock_size / BS_N;

    for(int k_tiled=0; k_tiled<k_TILED_MAX; k_tiled+=BS_K){
      // Load A and B into shared memory
      for(int n_unit_row=0; n_unit_row < BS_M; n_unit_row += tb_unit_A){
        As[n_unit_row + As_i][As_j] = A[(n_unit_row + As_i)*K + As_j];
      }
      for(int n_unit_col=0; n_unit_col < BS_K; n_unit_col += tb_unit_B){
        Bs[n_unit_col + Bs_i][Bs_j] = B[(n_unit_col + Bs_i)*N + Bs_j];
      }
      __syncthreads();

      // update A and B position for next iteration
      A += BS_K;
      B += BS_K * N;

      // Compute partial results
      for(int k=0; k<BS_K; ++k){
        // block into registers
        for(int i=0; i<pT_M; ++i){
          temp_reg_A[i] = As[inblock_i*pT_M + i][k];
        }
        for(int j=0; j<pT_N; ++j){
          temp_reg_B[j] = Bs[k][inblock_j*pT_N + j];
        }
        for(int i=0; i<pT_M; ++i){
          for(int j=0; j<pT_N; ++j){
            // per_thread_items[i*pT_N + j] += temp_reg_A[i] * temp_reg_B[j];
            per_thread_items[i*pT_N + j] += temp_reg_A[i] * temp_reg_B[j];
          }
        }
      }
      __syncthreads();
    }

    // perform update for untiled region(K)
    for(int k=k_TILED_MAX; k<K; k++){
      for(int i=0; i<pT_M; i++){
        temp_reg_A[i] = orig_A[(i + pT_M*inblock_i + BS_M*block_i)*K + k];
      }
      for(int j=0; j<pT_N; j++){
        temp_reg_B[j] = orig_B[k*N + (j + pT_N*inblock_j + BS_N*block_j)];
      }
      for(int i=0; i<pT_M; i++){
        for(int j=0; j<pT_N; j++){
          // per_thread_items[i*pT_N + j] += temp_reg_A[i] * temp_reg_B[j];
          per_thread_items[i*pT_N + j] += temp_reg_A[i] * temp_reg_B[j];
        }
      }
    }

    // copy back result to C
    for(int i=0; i<pT_M; i++){
      for(int j=0; j<pT_N; j++){
        C[(i + pT_M*inblock_i)*N + j + pT_N*inblock_j] = per_thread_items[i*pT_N + j];
      }
    }
  }
  else // deal with leftover region
  {
    int global_i = block_i * BS_M + pT_M * inblock_i;
    int global_j = block_j * BS_N + pT_N * inblock_j;

    if(global_i >= M || global_j >= N) return; //check boundary condition

    for(int i=0; i<pT_M; i++){
      int global_i_iter = global_i + i;
      if(global_i_iter >= M) break; 

      for(int j=0; j<pT_N; j++){
        int global_j_iter = global_j + j;
        if(global_j_iter >= N) break;

        float sum = 0.0;
        for(int k=0; k<K; k++){
          sum += orig_A[global_i_iter*K + k] * orig_B[k*N + global_j_iter];
        }

        orig_C[global_i_iter*N + global_j_iter] = sum;
      }
    }
  }
}


__global__ void matmul_kernel_iter(float *A, float *B, float *C, int M, int N, int K, size_t token_num) {

  // hyperparameters to tune(start), keep this value same as the one in launch code(below)
  const unsigned int BS_M = 32;
  const unsigned int BS_N = 64;
  const unsigned int BS_K = 16;
  const unsigned int pT_M = 8;
  const unsigned int pT_N = 4;
  // hyperparameters to tune(end)

  const unsigned int k_TILED_MAX = (K/BS_K)*BS_K;
  const unsigned int N_TILE_i = M / BS_M;
  const unsigned int N_TILE_j = N / BS_N;
  const unsigned int threadblock_size = (BS_M*BS_N) / (pT_M*pT_N);

  int block_i = blockIdx.y;
  int block_j = blockIdx.x;
  int inblock_i = threadIdx.x / (BS_N/pT_N);
  int inblock_j = threadIdx.x % (BS_N/pT_N);

  // save the original pointers for return
  float * orig_A = A;
  float * orig_B = B;
  float * orig_C = C;

  // shared memory for A and B
  __shared__ float As[BS_M][BS_K];
  __shared__ float Bs[BS_K][BS_N];

  float per_thread_items[pT_M*pT_N] = {0};
  float temp_reg_A[pT_M] = {0};
  float temp_reg_B[pT_N] = {0};

  if(block_i < N_TILE_i && block_j < N_TILE_j)
  {
    // loop-invariant code motion
    A += (block_i * BS_M) * K;
    B += block_j * BS_N;
    C += (block_i * BS_M) * N + block_j * BS_N;

    const unsigned int As_j = threadIdx.x % BS_K;
    const unsigned int As_i = threadIdx.x / BS_K;
    const unsigned int Bs_j = threadIdx.x % BS_N;
    const unsigned int Bs_i = threadIdx.x / BS_N;
    const unsigned int tb_unit_A = threadblock_size / BS_K;
    const unsigned int tb_unit_B = threadblock_size / BS_N;

    for(int k_tiled=0; k_tiled<k_TILED_MAX; k_tiled+=BS_K){
      // Load A and B into shared memory
      for(int n_unit_row=0; n_unit_row < BS_M; n_unit_row += tb_unit_A){
        As[n_unit_row + As_i][As_j] = A[(n_unit_row + As_i)*K + As_j];
      }
      for(int n_unit_col=0; n_unit_col < BS_K; n_unit_col += tb_unit_B){
        Bs[n_unit_col + Bs_i][Bs_j] = B[(n_unit_col + Bs_i)*N + Bs_j];
      }
      __syncthreads();

      // update A and B position for next iteration
      A += BS_K;
      B += BS_K * N;

      // Compute partial results
      for(int k=0; k<BS_K; ++k){
        // block into registers
        for(int i=0; i<pT_M; ++i){
          temp_reg_A[i] = As[inblock_i*pT_M + i][k];
        }
        for(int j=0; j<pT_N; ++j){
          temp_reg_B[j] = Bs[k][inblock_j*pT_N + j];
        }
        for(int i=0; i<pT_M; ++i){
          for(int j=0; j<pT_N; ++j){
            // per_thread_items[i*pT_N + j] += temp_reg_A[i] * temp_reg_B[j];
            per_thread_items[i*pT_N + j] += temp_reg_A[i] * temp_reg_B[j];
          }
        }
      }
      __syncthreads();
    }

    // perform update for untiled region(K)
    for(int k=k_TILED_MAX; k<K; k++){
      for(int i=0; i<pT_M; i++){
        temp_reg_A[i] = orig_A[(i + pT_M*inblock_i + BS_M*block_i)*K + k];
      }
      for(int j=0; j<pT_N; j++){
        temp_reg_B[j] = orig_B[k*N + (j + pT_N*inblock_j + BS_N*block_j)];
      }
      for(int i=0; i<pT_M; i++){
        for(int j=0; j<pT_N; j++){
          // per_thread_items[i*pT_N + j] += temp_reg_A[i] * temp_reg_B[j];
          per_thread_items[i*pT_N + j] += temp_reg_A[i] * temp_reg_B[j];
        }
      }
    }

    // copy back result to C
    for(int i=0; i<pT_M; i++){
      for(int j=0; j<pT_N; j++){
        C[(i + pT_M*inblock_i)*N + j + pT_N*inblock_j] = per_thread_items[i*pT_N + j];
      }
    }
  }
  else // deal with leftover region
  {
    int global_i = block_i * BS_M + pT_M * inblock_i;
    int global_j = block_j * BS_N + pT_N * inblock_j;

    if(global_i >= M || global_j >= N) return; //check boundary condition

    for(int i=0; i<pT_M; i++){
      int global_i_iter = global_i + i;
      if(global_i_iter >= M) break; 

      for(int j=0; j<pT_N; j++){
        int global_j_iter = global_j + j;
        if(global_j_iter >= N) break;

        float sum = 0.0;
        for(int k=0; k<K; k++){
          sum += orig_A[global_i_iter*K + k] * orig_B[k*N + global_j_iter];
        }

        orig_C[global_i_iter*N + global_j_iter] = sum;
      }
    }
  }
}

/* Matmul_batch_final + cuda
 * @param [in1]  in1: [b, M, K]
 * @param [in2]  in2: [K, N] // wte_transposed_a
 * @param [out]  out: [b, M, N]
 */
void matmul_batch_final_cuda(Tensor *in1, Tensor *in2, Tensor *out, size_t token_num) {

  if (token_num == 0) {
    // hyperparameters to tune(start)
    const unsigned int BS_M = 32;
    const unsigned int BS_N = 64;
    // const unsigned int BS_K = 16;
    const unsigned int pT_M = 8;
    const unsigned int pT_N = 4;
    // hyperparameters to tune(end)

    size_t M = in1->shape[1];
    size_t K = in1->shape[2];
    size_t N = in2->shape[1];

    size_t n_batches = in1->shape[0];

    // kernel launch
    dim3 blockDim((BS_N * BS_M)/(pT_N * pT_M));
    dim3 gridDim(ROUND_DIV(N, BS_N), ROUND_DIV(n_batches * M, BS_M));
    matmul_kernel<<<gridDim, blockDim>>>(in1->cuda_buf, in2->cuda_buf, out->cuda_buf, (n_batches * M), N, K, token_num);

    CHECK_CUDA(cudaGetLastError());
  }
  else {
    // hyperparameters to tune(start)
    const unsigned int BS_M = 32;
    const unsigned int BS_N = 64;
    // const unsigned int BS_K = 16;
    const unsigned int pT_M = 8;
    const unsigned int pT_N = 4;
    // hyperparameters to tune(end)

    size_t M = in1->shape[1];
    size_t K = in1->shape[2];
    size_t N = in2->shape[1];

    size_t n_batches = in1->shape[0];

    // kernel launch
    dim3 blockDim((BS_N * BS_M)/(pT_N * pT_M));
    dim3 gridDim(ROUND_DIV(N, BS_N), ROUND_DIV(n_batches * M, BS_M));
    matmul_kernel<<<gridDim, blockDim>>>(in1->cuda_buf, in2->cuda_buf, out->cuda_buf, (n_batches * M), N, K, token_num);

    CHECK_CUDA(cudaGetLastError());
  }
}



/////////////////////////////* 10. add *///////////////////////////// 
__global__ void add_kernel_batch(float *inout, float *x, size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) { inout[idx] += x[idx]; }
}

void add_batch_cuda(Tensor *inout, Tensor *x) {
  size_t N = inout->num_elem();

  const unsigned int BLOCK_SIZE = 256;
  dim3 blockDim(BLOCK_SIZE);
  dim3 gridDim(ROUND_DIV(N, BLOCK_SIZE));

  add_kernel_batch<<<gridDim, blockDim>>>(inout->cuda_buf, x->cuda_buf, N);
}



/////////////////////////////* 11. copy *///////////////////////////// 
__global__ void copy_kernel_batch(float *in, float *out, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    out[idx] = in[idx];
  }
}

void copy_batch_cuda(Tensor *in, Tensor *out) {
  size_t N = in->num_elem();

  // define grid and block size
  const unsigned int BLOCK_SIZE = 256;
  dim3 gridDim(ROUND_DIV(N, BLOCK_SIZE));
  dim3 blockDim(BLOCK_SIZE);

  // call kernel
  copy_kernel_batch<<<gridDim, blockDim>>>(in->cuda_buf, out->cuda_buf, N);
}



/////////////////////////////* 12. split_qkv */////////////////////////////
__global__ void split_qkv_kernel_batch(float *in, float *out, int s, int H) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int batch_idx = blockIdx.y;

  in += batch_idx * (s * H);
  out += batch_idx * (s * H);

  if (idx < s * H) {
    size_t i = idx / (s * (H / 3));
    size_t j = (idx % (s * (H / 3))) / (H / 3);
    size_t k = idx % (H / 3);

    out[i * s * (H / 3) + j * (H / 3) + k] = in[i * (H / 3) + j * 3 * (H / 3) + k];
  }
}

/* Split into QKV batch
 * @param [in1]  in: [b, s, H]
 * @param [out] out: [b, 3, s, H/3]
 */
void split_qkv_batch_cuda(Tensor *in, Tensor *out) {
  size_t s = in->shape[1];
  size_t H = in->shape[2];

  size_t n_batches = in->shape[0];

  // define grid and block size
  const unsigned int BLOCK_SIZE = 256;
  dim3 gridDim(ROUND_DIV(s * H, BLOCK_SIZE), n_batches);
  dim3 blockDim(BLOCK_SIZE);

  // call kernel
  split_qkv_kernel_batch<<<gridDim, blockDim>>>(in->cuda_buf, out->cuda_buf, s, H);
}



/////////////////////////////* 13. split_head */////////////////////////////
__global__ void split_head_kernel_batch(float *in, float *out, int s, int H, int n_head) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int batch_idx = blockIdx.y;

  in += batch_idx * (3 * s * H);
  out += batch_idx * (3 * s * H);

  if (idx < 3 * s * H) {
    size_t i = idx / (s * H);
    size_t j = (idx % (s * H)) / (s * (H / n_head));
    size_t k = (idx % (s * (H / n_head))) / (H / n_head);
    size_t l = idx % (H / n_head);

    out[i * (n_head * s * (H / n_head)) + j * (s * (H / n_head)) + k * (H / n_head) + l] = in[i * (s * H) + k * H + j * (H / n_head) + l];
  }
}


/* Split into heads batch + cuda
 * @param [in1]  in: [b, 3, s, H]
 * @param [out] out: [b, 3, n_head, s, H/n_head]
 * 's' is the number of tokens in the prompt.
 * 'H' is the hidden dimension.
 * 'n_head' is the number of heads.
 */
void split_head_batch_cuda(Tensor *in, size_t n_head, Tensor *out) {
  size_t s = in->shape[2];
  size_t H = in->shape[3];

  size_t n_batches = in->shape[0];

  size_t tensor_size = in->num_elem();

  // define grid and block size
  const unsigned int BLOCK_SIZE = 256;
  dim3 gridDim(ROUND_DIV(3 * s * H, BLOCK_SIZE), n_batches);
  dim3 blockDim(BLOCK_SIZE);

  // call kernel
  split_head_kernel_batch<<<gridDim, blockDim>>>(in->cuda_buf, out->cuda_buf, s, H, n_head);
}



/////////////////////////////* 14. generate mask */////////////////////////////
__global__ void generate_mask_kernel_batch(float *mask, int s) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int batch_idx = blockIdx.y;

  mask += batch_idx * s * s;

  if (idx < s * s) {
    int i = idx / s;
    int j = idx % s;
    if (i >= j) {
      mask[i * s + j] = 0;
    } else {
      mask[i * s + j] = -1e10;
    }
  }
}

/* Generate mask batch + cuda
 * @param [in & out] inout: [b, s, s] -> [b, 1, s] (for tokens after the prompt phase)
 * 's' is the number of tokens in the prompt.
 */
void generate_mask_batch_cuda(Tensor *inout, size_t token_num) {
  if (token_num == 0) {
    size_t s = inout->shape[1];
    size_t n_batches = inout->shape[0];

    // define grid and block size
    const unsigned int BLOCK_SIZE = 256;
    dim3 gridDim(ROUND_DIV(s * s, BLOCK_SIZE), n_batches);
    dim3 blockDim(BLOCK_SIZE);

    // call kernel
    generate_mask_kernel_batch<<<gridDim, blockDim>>>(inout->cuda_buf, s);
  }
  else {
    size_t s = inout->shape[2];
    size_t n_batches = inout->shape[0];
    float *zero_board = (float *)calloc(s * n_batches, sizeof(float));
    CHECK_CUDA(cudaMemcpy(inout->cuda_buf, zero_board, s * n_batches * sizeof(float), cudaMemcpyHostToDevice));
  }
}



/////////////////////////////* 15. concat head */////////////////////////////
__global__ void concat_head_kernel_batch(float *in, float *out, int s, int H, int n_head) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int batch_idx = blockIdx.y;

  in += batch_idx * (n_head * s * H);
  out += batch_idx * (n_head * s * H);

  if (idx < n_head * s * H) {
    size_t i = idx / (H * n_head);
    size_t j = (idx % (H * n_head)) / H;
    size_t k = idx % H;

    out[i * (n_head * H) + j * H + k] = in[j * (s * H) + i * H + k];
  }
}

/* Concatenate each heads batch + cuda
 * @param [in1]     in: [b, n_head, s, H_]
 * @param [out]    out: [b, s, H_*n_head]
 * 'n_head' is the number of heads.
 * 's' is the number of tokens in the prompt.
 * 'H_' is the hidden dimension/n_head.
 */
void concat_head_batch_cuda(Tensor *in, Tensor *out) {
  size_t n_head = in->shape[1];
  size_t s = in->shape[2];
  size_t H = in->shape[3];

  size_t n_batches = in->shape[0];

  size_t in_size = in->num_elem();
  size_t out_size = out->num_elem();

  // define grid and block size
  const unsigned int BLOCK_SIZE = 256;
  dim3 gridDim(ROUND_DIV(s * H * n_head, BLOCK_SIZE), n_batches);
  dim3 blockDim(BLOCK_SIZE);

  // call kernel
  concat_head_kernel_batch<<<gridDim, blockDim>>>(in->cuda_buf, out->cuda_buf, s, H, n_head);
}



/////////////////////////////* 16. extract qkv */////////////////////////////
__global__ void extract_qkv_kernel_batch(float *in, float *q, float *k, float *v, int s, int H, int head_idx, int n_head) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int batch_idx = blockIdx.y;

  in += batch_idx * (3 * n_head * s * H);
  q += batch_idx * (s * H);
  k += batch_idx * (s * H);
  v += batch_idx * (s * H);

  int i = idx / H;
  int j = idx % H;

  if (idx < s * H) {
    q[i * H + j] = in[0 * n_head * s * H + head_idx * s * H + i * H + j];
    k[i * H + j] = in[1 * n_head * s * H + head_idx * s * H + i * H + j];
    v[i * H + j] = in[2 * n_head * s * H + head_idx * s * H + i * H + j];
  }
}

/* Extract Q, K, V from QKV head batch + cuda
 * @param [in1]       in: [b, 3, n_head, s, H_]
 * @param [in2] head_idx: [1]
 * @param [in3]   n_head: [1]
 * @param [out]        q: [b, s, H_]
 * @param [out]        k: [b, s, H_]
 * @param [out]        v: [b, s, H_]
 * 's' is the number of tokens in the prompt.
 * 'H_' is the hidden dimension/n_head.
 * 'n_head' is the number of heads.
 */
void extract_qkv_batch_cuda(Tensor *in, size_t head_idx, size_t n_head, Tensor *q, Tensor *k, Tensor *v) {
  size_t s = in->shape[3];
  size_t H = in->shape[4];

  size_t n_batches = in->shape[0];

  size_t in_size = in->num_elem();
  size_t qkv_size = q->num_elem();

  // define grid and block size
  const unsigned int BLOCK_SIZE = 256;
  dim3 gridDim(ROUND_DIV(s * H, BLOCK_SIZE), n_batches);
  dim3 blockDim(BLOCK_SIZE);

  // call kernel
  extract_qkv_kernel_batch<<<gridDim, blockDim>>>(in->cuda_buf, q->cuda_buf, k->cuda_buf, v->cuda_buf, s, H, head_idx, n_head);
}



/////////////////////////////* 17. merge head */////////////////////////////
__global__ void merge_head_kernel_batch(float *in, float *out, int s, int H, int haed_idx, int n_head) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int batch_idx = blockIdx.y;

  in += batch_idx * (s * H);
  out += batch_idx * (n_head * s * H);

  if (idx < s * H) {
    size_t i = idx / H;
    size_t j = idx % H;

    out[haed_idx * s * H + i * H + j] = in[i * H + j];
  }
}

/* Merge each heads batch + cuda
 * @param [in1]       in: [b, s, H_]
 * @param [in2] head_idx: [1]
 * @param [in3]   n_head: [1]
 * @param [out]      out: [b, n_head, s, H_]
 * 's' is the number of tokens in the prompt.
 * 'H_' is the hidden dimension/n_head.
 * 'n_head' is the number of heads.
 */
void merge_head_batch_cuda(Tensor *in, size_t head_idx, size_t n_head, Tensor *out) {
  size_t s = in->shape[1];
  size_t H = in->shape[2];

  size_t num_batches = in->shape[0];

  size_t in_size = in->num_elem();
  size_t out_size = out->num_elem();

  // define grid and block size
  const unsigned int BLOCK_SIZE = 256;
  dim3 gridDim(ROUND_DIV(s * H, BLOCK_SIZE), num_batches);
  dim3 blockDim(BLOCK_SIZE);

  // call kernel
  merge_head_kernel_batch<<<gridDim, blockDim>>>(in->cuda_buf, out->cuda_buf, s, H, head_idx, n_head);
}



/////////////////////////////* 18. select_last_token_cuda */////////////////////////////
__global__ void select_last_token_kernel(float *in, float *out, int s, int H) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int batch_idx = blockIdx.y;

  in += batch_idx * (s * H);
  out += batch_idx * H;

  if (idx < H) {
    out[idx] = in[(s - 1) * H + idx];
  }
}

/* Select only the last token to reduce unnecessary computation & data movement
 * @param [in]       in: [b, s, H]
 * @param [out]     out: [b, 1, H]
 * 's' is the number of tokens in the prompt.
 * 'H' is the hidden dimension.
*/
void select_last_token_cuda(Tensor *in, Tensor *out) {
  size_t s = in->shape[1];
  size_t H = in->shape[2];

  size_t n_batches = in->shape[0];

  // define grid and block size
  const unsigned int BLOCK_SIZE = 256;
  dim3 gridDim(ROUND_DIV(H, BLOCK_SIZE), n_batches);
  dim3 blockDim(BLOCK_SIZE);

  // call kernel
  select_last_token_kernel<<<gridDim, blockDim>>>(in->cuda_buf, out->cuda_buf, s, H);
}



/////////////////////////////* 19. insert_kv_cache */////////////////////////////
const size_t NUM_HEAD = 12;
// const size_t NUM_LAYER = 12;

__global__ void insert_kv_kernel(float *k_cache, float *v_cache, float *k, float *v, size_t layer_idx, size_t head_idx, size_t n_batches, size_t s, size_t H) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int batch_idx = blockIdx.y;

  k_cache += (layer_idx * NUM_HEAD * n_batches * 24 * (H) + head_idx * n_batches * 24 * (H) + batch_idx * 24 * (H));
  v_cache += (layer_idx * NUM_HEAD * n_batches * 24 * (H) + head_idx * n_batches * 24 * (H) + batch_idx * 24 * (H));

  k += batch_idx * s * (H);
  v += batch_idx * s * (H);

  if (idx < s * (H)) {
    k[idx] = k_cache[idx];
    v[idx] = v_cache[idx];
  }
}


/* Insert stored KV cache
 * @param [k_cache]       k_cache: [NUM_LAYER, NUM_HEAD, n_batches, 24, HIDDEN_DIM / NUM_HEAD]
 * @param [v_cache]       v_cache: [NUM_LAYER, NUM_HEAD, n_batches, 24, HIDDEN_DIM / NUM_HEAD]
 * @param [k]                   k: [n_batches, prompt_size, HIDDEN_DIM / NUM_HEAD]
 * @param [v]                   v: [n_batches, prompt_size, HIDDEN_DIM / NUM_HEAD]
 * 
 * move cached kv value to k and v
 * 
*/
void insert_kv_cache(Tensor *k_cache, Tensor *v_cache, Tensor *k, Tensor *v, size_t layer_idx, size_t head_idx, size_t token_num) {
  size_t n_batches = k_cache->shape[2];
  size_t s = k->shape[1]; // 'prompt_size' for prompt phase
  size_t H = k_cache->shape[4]; // 'HIDDEN_DIM / NUM_HEAD'

  // define grid and block size
  const unsigned int BLOCK_SIZE = 64;
  dim3 gridDim(ROUND_DIV(s * H, BLOCK_SIZE), n_batches);
  dim3 blockDim(BLOCK_SIZE);

  // call kernel
  insert_kv_kernel<<<gridDim, blockDim>>>(k_cache->cuda_buf, v_cache->cuda_buf, k->cuda_buf, v->cuda_buf, layer_idx, head_idx, n_batches, s, H);
}



/////////////////////////////* 20. store_kv_cache */////////////////////////////
__global__ void store_kv_cache_kernel(float *k_cache, float *v_cache, float *k_temp, float *v_temp, size_t layer_idx, size_t head_idx, size_t n_batches, size_t s, size_t H) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int batch_idx = blockIdx.y;

  k_cache += (layer_idx * NUM_HEAD * n_batches * 24 * (H) + head_idx * n_batches * 24 * (H) + batch_idx * 24 * (H));
  v_cache += (layer_idx * NUM_HEAD * n_batches * 24 * (H) + head_idx * n_batches * 24 * (H) + batch_idx * 24 * (H));

  k_temp += batch_idx * s * (H);
  v_temp += batch_idx * s * (H);

  if (idx < s * (H)) {
    k_cache[idx] = k_temp[idx];
    v_cache[idx] = v_temp[idx];
  }
}

__global__ void store_kv_cache_kernel_iter(float *k_cache, float *v_cache, float *k_temp, float *v_temp, size_t layer_idx, size_t head_idx, size_t n_batches, size_t token_num, size_t H) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n_batches * H) {
    int batch_idx = idx / H;
    int j = idx % H;

    k_cache += (layer_idx * NUM_HEAD * n_batches * 24 * (H) + head_idx * n_batches * 24 * (H) + batch_idx * 24 * (H));
    v_cache += (layer_idx * NUM_HEAD * n_batches * 24 * (H) + head_idx * n_batches * 24 * (H) + batch_idx * 24 * (H));

    k_cache += (token_num + 15) * (H);
    v_cache += (token_num + 15) * (H);

    k_temp += batch_idx * H;
    v_temp += batch_idx * H;

    k_cache[j] = k_temp[j];
    v_cache[j] = v_temp[j];
  }
}


/* Store generated k and v to kv_cache
 * @param [k_cache]       k_cache: [NUM_LAYER, NUM_HEAD, n_batches, 24, HIDDEN_DIM / NUM_HEAD]
 * @param [v_cache]       v_cache: [NUM_LAYER, NUM_HEAD, n_batches, 24, HIDDEN_DIM / NUM_HEAD]
 * @param [k]              k_temp: [n_batches, 1 or prompt_size, HIDDEN_DIM / NUM_HEAD]
 * @param [v]              v_temp: [n_batches, 1 or prompt_size, HIDDEN_DIM / NUM_HEAD]
 * 
 * store generated k and v to kv_cache
 * 
*/
void store_kv_cache(Tensor *k_cache, Tensor *v_cache, Tensor *k_temp, Tensor *v_temp, size_t layer_idx, size_t head_idx, size_t token_num) {
  if (token_num == 0) {
    size_t n_batches = k_cache->shape[2];
    size_t s = k_temp->shape[1]; // 'prompt_size' for prompt phase
    size_t H = k_temp->shape[2]; // 'HIDDEN_DIM / NUM_HEAD'

    // define grid and block size
    const unsigned int BLOCK_SIZE = 64;
    dim3 gridDim(ROUND_DIV(s * H, BLOCK_SIZE), n_batches);
    dim3 blockDim(BLOCK_SIZE);

    // call kernel
    store_kv_cache_kernel<<<gridDim, blockDim>>>(k_cache->cuda_buf, v_cache->cuda_buf, k_temp->cuda_buf, v_temp->cuda_buf, layer_idx, head_idx, n_batches, s, H);
  }
  else {
    size_t n_batches = k_cache->shape[2];
    size_t H = k_temp->shape[2]; // 'HIDDEN_DIM / NUM_HEAD'

    // define grid and block size
    const unsigned int BLOCK_SIZE = 256;
    dim3 gridDim(ROUND_DIV(n_batches * H, BLOCK_SIZE));
    dim3 blockDim(BLOCK_SIZE);

    // call kernel
    store_kv_cache_kernel_iter<<<gridDim, blockDim>>>(k_cache->cuda_buf, v_cache->cuda_buf, k_temp->cuda_buf, v_temp->cuda_buf, layer_idx, head_idx, n_batches, token_num, H);
  }
}