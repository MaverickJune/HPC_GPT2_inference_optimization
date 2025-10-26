#pragma once

#include <vector>

#include "tensor.h"

using std::vector;

#define MATH_PI 3.1415926535f

/* Elementwise operations */
void gelu(Tensor *inout);
void add(Tensor *inout, Tensor *x);
void add_cuda(Tensor *inout, Tensor *x);
void scaling(Tensor *inout, float scale);

/* Matmul operations */
void linear(Tensor *in, Tensor *w, Tensor *b, Tensor *out);
void matmul(Tensor *in1, Tensor *in2, Tensor *out);

/* Data movement operations */
void copy(Tensor *in, Tensor *out);
void transpose(Tensor *in, Tensor *out);
void split_qkv(Tensor *in, Tensor *out);
void split_head(Tensor *in, size_t n_head, Tensor *out);
void concat_head(Tensor *in, Tensor *out);
void extract_qkv(Tensor *in, size_t head_idx, size_t n_head, Tensor *q,
                 Tensor *k, Tensor *v);
void merge_head(Tensor *in, size_t head_idx, size_t n_head, Tensor *out);
void token_pos_embedding(vector<int> in, Parameter *wte, Parameter *wpe,
                         Tensor *out);

/* Other operations */
void softmax(Tensor *inout);
void layer_norm(Tensor *inout, Tensor *gamma, Tensor *beta);
void generate_mask(Tensor *inout);
int top1_sampling(Tensor *in);

/* batch version added */
void token_pos_embedding_batch(vector<int> in[], Parameter *wte, Parameter *wpe, Tensor *out);
void copy_batch(Tensor *in, Tensor *out);

void linear_batch(Tensor *in, Tensor *w, Tensor *b, Tensor *out);
void gelu_batch(Tensor *inout);

void transpose_batch(Tensor *in, Tensor *out);
void matmul_batch(Tensor *in1, Tensor *in2, Tensor *out);
void layer_norm_batch(Tensor *inout, Tensor *gamma, Tensor *beta);
void add_batch(Tensor *inout, Tensor *x);

void split_qkv_batch(Tensor *in, Tensor *out);
void split_head_batch(Tensor *in, size_t n_head, Tensor *out);
void generate_mask_batch(Tensor *inout);
void concat_head_batch(Tensor *in, Tensor *out);
void extract_qkv_batch(Tensor *in, size_t head_idx, size_t n_head, Tensor *q, Tensor *k, Tensor *v);
void merge_head_batch(Tensor *in, size_t head_idx, size_t n_head, Tensor *out);

void scaling_batch(Tensor *inout, float scale);
void softmax_batch(Tensor *inout);

vector<int> top1_sampling_batch(Tensor *in);

// added
void matmul_batch_final(Tensor *in1, Tensor *in2, Tensor *out);

/* batch + cuda version */
void matmul_batch_cuda(Tensor *in1, Tensor *in2, Tensor *out, size_t token_num); // pass
void linear_batch_cuda(Tensor *in, Tensor *w, Tensor *b, Tensor *out, size_t token_num); // pass
void token_pos_embedding_batch_cuda(vector<int> in[], Parameter *wte, Parameter *wpe, Tensor *out, size_t token_num); // pass
void gelu_batch_cuda(Tensor *inout); // pass
void layer_norm_batch_cuda(Tensor *inout, Tensor *gamma, Tensor *beta); // pass
void scaling_batch_cuda(Tensor *inout, float scale); // pass
void softmax_batch_cuda(Tensor *inout); // pass

void transpose_cuda(Tensor *in, Tensor *out); // pass
void transpose_batch_cuda(Tensor *in, Tensor *out); // pass
void matmul_batch_final_cuda(Tensor *in1, Tensor *in2, Tensor *out, size_t token_num); // pass

void add_batch_cuda(Tensor *inout, Tensor *x); // pass
void copy_batch_cuda(Tensor *in, Tensor *out); // pass

void split_qkv_batch_cuda(Tensor *in, Tensor *out); // pass
void split_head_batch_cuda(Tensor *in, size_t n_head, Tensor *out); // pass
void generate_mask_batch_cuda(Tensor *inout, size_t token_num); // pass
void concat_head_batch_cuda(Tensor *in, Tensor *out); // pass
void extract_qkv_batch_cuda(Tensor *in, size_t head_idx, size_t n_head, Tensor *q, Tensor *k, Tensor *v); // pass
void merge_head_batch_cuda(Tensor *in, size_t head_idx, size_t n_head, Tensor *out);

// added
void select_last_token_cuda(Tensor *in, Tensor *out);

// added for kv caching
void insert_kv_cache(Tensor *k_cache, Tensor *v_cache, Tensor *k, Tensor *v, size_t layer_idx, size_t head_idx, size_t token_num);
void store_kv_cache(Tensor *k_cache, Tensor *v_cache, Tensor *k_temp, Tensor *v_temp, size_t layer_idx, size_t head_idx, size_t token_num);

// added for D2H transfer optimization
// void top_1_sampling_cuda(Tensor *in, Tensor *out);