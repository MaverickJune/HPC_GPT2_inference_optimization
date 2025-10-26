#include <mpi.h>

#include <cmath>
#include <cstdio>

#include "layer.h"
#include "model.h"

#include <assert.h>

#define CHECK_CUDA(call)                                              \
  do {                                                                \
    cudaError_t status_ = call;                                       \
    if (status_ != cudaSuccess) {                                     \
      fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(status_));                           \
      exit(EXIT_FAILURE);                                             \
    }                                                                 \
  } while (0)


#define N_GPU 4

//////////////////// BATCH SIZE ////////////////////
const size_t BATCH_SIZE = 256;
////////////////////////////////////////////////////


/////////////////// CONSTANTS //////////////////////
const size_t num_generated_tokens = 8;
////////////////////////////////////////////////////


/* Activations (relocated for unified initialization) */
Activation *embd_a[N_GPU], *ffn_proj_a[N_GPU];
Activation *mha_qkv_proj_a[N_GPU], *mha_out_a[N_GPU], *mha_split_qkv_a[N_GPU], *mha_split_head_a[N_GPU],
    *mha_mask_a[N_GPU], *mha_merge_head_a[N_GPU], *mha_q_a[N_GPU], *mha_k_a[N_GPU], *mha_v_a[N_GPU],
    *mha_attn_out_a[N_GPU], *mha_concat_head_a[N_GPU];
Activation *attn_score_a[N_GPU], *k_transposed_a[N_GPU];
Activation *wte_transposed_a[N_GPU], *residual_a[N_GPU], *logit_a[N_GPU];
Activation *transformer_block_a[N_GPU];

// added
Activation *embd_a_last[N_GPU];

// added for kv caching
Activation *k_cache_a[N_GPU], *v_cache_a[N_GPU];

// temporary k and v before inserting kv cache into them
Activation *k_temp_a[N_GPU], *v_temp_a[N_GPU];

/* added for D2H optimization */
// Activation *current_token_batch_gpu[N_GPU]; // [b, 1]
// Activation *output_gpu[N_GPU]; // [b, 8(output_len)]


void alloc_activations_batch_multi_gpu(size_t prompt_size, size_t n_batches, size_t idx) {
  embd_a[idx] = new Activation({n_batches, prompt_size, HIDDEN_DIM});

  ffn_proj_a[idx] = new Activation({n_batches, prompt_size, 4 * HIDDEN_DIM});

  mha_qkv_proj_a[idx] = new Activation({n_batches, prompt_size, 3 * HIDDEN_DIM});
  mha_out_a[idx] = new Activation({n_batches, prompt_size, HIDDEN_DIM});
  mha_split_qkv_a[idx] = new Activation({n_batches, 3, prompt_size, HIDDEN_DIM});
  mha_split_head_a[idx] =
      new Activation({n_batches, 3, NUM_HEAD, prompt_size, HIDDEN_DIM / NUM_HEAD});
  mha_mask_a[idx] = new Activation({n_batches, prompt_size, prompt_size});
  mha_merge_head_a[idx] = new Activation({n_batches, NUM_HEAD, prompt_size, HIDDEN_DIM / NUM_HEAD});
  mha_q_a[idx] = new Activation({n_batches, prompt_size, HIDDEN_DIM / NUM_HEAD});
  mha_k_a[idx] = new Activation({n_batches, prompt_size, HIDDEN_DIM / NUM_HEAD});
  mha_v_a[idx] = new Activation({n_batches, prompt_size, HIDDEN_DIM / NUM_HEAD});
  mha_attn_out_a[idx] = new Activation({n_batches, prompt_size, HIDDEN_DIM / NUM_HEAD});
  mha_concat_head_a[idx] = new Activation({n_batches, prompt_size, HIDDEN_DIM});

  attn_score_a[idx] = new Activation({n_batches, prompt_size, prompt_size});
  k_transposed_a[idx] = new Activation({n_batches, HIDDEN_DIM / NUM_HEAD, prompt_size});

  wte_transposed_a[idx] = new Activation({HIDDEN_DIM, NUM_VOCAB});

  residual_a[idx] = new Activation({n_batches, prompt_size, HIDDEN_DIM});
  // logit_a[idx] = new Activation({n_batches, prompt_size, NUM_VOCAB});
  logit_a[idx] = new Activation({n_batches, 1, NUM_VOCAB}); // select only the last token from each prompt in a batch
  transformer_block_a[idx] = new Activation({n_batches, prompt_size, HIDDEN_DIM});

  // added
  embd_a_last[idx] = new Activation({n_batches, 1, HIDDEN_DIM}); // select only the last token from each prompt in a batch

  // added for kv caching
  /* allocate space for 24 tokens per each attention head */
  k_cache_a[idx] = new Activation({NUM_LAYER, NUM_HEAD, n_batches, prompt_size, HIDDEN_DIM / NUM_HEAD});
  v_cache_a[idx] = new Activation({NUM_LAYER, NUM_HEAD, n_batches, prompt_size, HIDDEN_DIM / NUM_HEAD});

  // temporary k and v before inserting kv cache into them
  k_temp_a[idx] = new Activation({n_batches, prompt_size, HIDDEN_DIM / NUM_HEAD});
  v_temp_a[idx] = new Activation({n_batches, prompt_size, HIDDEN_DIM / NUM_HEAD});

  // added for D2H optimization
  // current_token_batch_gpu[idx] = new Activation({n_batches, 1});
  // output_gpu[idx] = new Activation({n_batches, num_generated_tokens});
}


void resize_activations_batch_multi_gpu(size_t prompt_size, size_t n_batches, size_t idx, size_t token_num) {
  if (token_num == 0) {
    /* deal with the first token */
    embd_a[idx]->resize_activation({n_batches, prompt_size, HIDDEN_DIM});

    ffn_proj_a[idx]->resize_activation({n_batches, prompt_size, 4 * HIDDEN_DIM});

    mha_qkv_proj_a[idx]->resize_activation({n_batches, prompt_size, 3 * HIDDEN_DIM});
    mha_out_a[idx]->resize_activation({n_batches, prompt_size, HIDDEN_DIM});
    mha_split_qkv_a[idx]->resize_activation({n_batches, 3, prompt_size, HIDDEN_DIM});
    mha_split_head_a[idx]->resize_activation({n_batches, 3, NUM_HEAD, prompt_size, HIDDEN_DIM / NUM_HEAD});
    mha_mask_a[idx]->resize_activation({n_batches, prompt_size, prompt_size});
    mha_merge_head_a[idx]->resize_activation({n_batches, NUM_HEAD, prompt_size, HIDDEN_DIM / NUM_HEAD});
    mha_q_a[idx]->resize_activation({n_batches, prompt_size, HIDDEN_DIM / NUM_HEAD});
    mha_k_a[idx]->resize_activation({n_batches, prompt_size, HIDDEN_DIM / NUM_HEAD});
    mha_v_a[idx]->resize_activation({n_batches, prompt_size, HIDDEN_DIM / NUM_HEAD});
    mha_attn_out_a[idx]->resize_activation({n_batches, prompt_size, HIDDEN_DIM / NUM_HEAD});
    mha_concat_head_a[idx]->resize_activation({n_batches, prompt_size, HIDDEN_DIM});

    attn_score_a[idx]->resize_activation({n_batches, prompt_size, prompt_size});
    k_transposed_a[idx]->resize_activation({n_batches, HIDDEN_DIM / NUM_HEAD, prompt_size});

    wte_transposed_a[idx]->resize_activation({HIDDEN_DIM, NUM_VOCAB});

    residual_a[idx]->resize_activation({n_batches, prompt_size, HIDDEN_DIM});
    transformer_block_a[idx]->resize_activation({n_batches, prompt_size, HIDDEN_DIM});

    k_temp_a[idx]->resize_activation({n_batches, prompt_size, HIDDEN_DIM / NUM_HEAD});
    v_temp_a[idx]->resize_activation({n_batches, prompt_size, HIDDEN_DIM / NUM_HEAD});
  }
  else {
    /* after the first token */
    embd_a[idx]->resize_activation({n_batches, 1, HIDDEN_DIM});

    ffn_proj_a[idx]->resize_activation({n_batches, 1, 4 * HIDDEN_DIM});

    mha_qkv_proj_a[idx]->resize_activation({n_batches, 1, 3 * HIDDEN_DIM});
    mha_out_a[idx]->resize_activation({n_batches, 1, HIDDEN_DIM});
    mha_split_qkv_a[idx]->resize_activation({n_batches, 3, 1, HIDDEN_DIM});
    mha_split_head_a[idx]->resize_activation({n_batches, 3, NUM_HEAD, 1, HIDDEN_DIM / NUM_HEAD});
    mha_mask_a[idx]->resize_activation({n_batches, 1, prompt_size}); // will not use mask at autoregressive-generation phase
    mha_merge_head_a[idx]->resize_activation({n_batches, NUM_HEAD, 1, HIDDEN_DIM / NUM_HEAD});
    mha_q_a[idx]->resize_activation({n_batches, 1, HIDDEN_DIM / NUM_HEAD});
    mha_k_a[idx]->resize_activation({n_batches, prompt_size, HIDDEN_DIM / NUM_HEAD}); // k cache size should be equal to prompt
    mha_v_a[idx]->resize_activation({n_batches, prompt_size, HIDDEN_DIM / NUM_HEAD}); // v cache size should be equal to prompt
    mha_attn_out_a[idx]->resize_activation({n_batches, 1, HIDDEN_DIM / NUM_HEAD});
    mha_concat_head_a[idx]->resize_activation({n_batches, 1, HIDDEN_DIM});

    attn_score_a[idx]->resize_activation({n_batches, 1, prompt_size});
    k_transposed_a[idx]->resize_activation({n_batches, HIDDEN_DIM / NUM_HEAD, prompt_size});

    wte_transposed_a[idx]->resize_activation({HIDDEN_DIM, NUM_VOCAB});

    residual_a[idx]->resize_activation({n_batches, 1, HIDDEN_DIM});
    transformer_block_a[idx]->resize_activation({n_batches, 1, HIDDEN_DIM});

    k_temp_a[idx]->resize_activation({n_batches, 1, HIDDEN_DIM / NUM_HEAD});
    v_temp_a[idx]->resize_activation({n_batches, 1, HIDDEN_DIM / NUM_HEAD});
  }
}

void free_activations_multi_gpu(size_t idx) {
  delete embd_a[idx];
  delete ffn_proj_a[idx];
  delete mha_qkv_proj_a[idx];
  delete mha_out_a[idx];
  delete mha_split_qkv_a[idx];
  delete mha_split_head_a[idx];
  delete mha_mask_a[idx];
  delete mha_merge_head_a[idx];
  delete mha_q_a[idx];
  delete mha_k_a[idx];
  delete mha_v_a[idx];
  delete mha_attn_out_a[idx];
  delete mha_concat_head_a[idx];
  delete attn_score_a[idx];
  delete k_transposed_a[idx];
  delete wte_transposed_a[idx];
  delete residual_a[idx];
  delete logit_a[idx];
  delete transformer_block_a[idx];

  // added
  delete embd_a_last[idx];

  // added for kv caching
  delete k_cache_a[idx];
  delete v_cache_a[idx];

  delete k_temp_a[idx];
  delete v_temp_a[idx];
}


/* Parameters (relocated for unified initialization) */
Parameter *attn_b[N_GPU][NUM_LAYER], *attn_w[N_GPU][NUM_LAYER];
Parameter *proj_b[N_GPU][NUM_LAYER], *proj_w[N_GPU][NUM_LAYER];
Parameter *ln_1_b[N_GPU][NUM_LAYER], *ln_1_g[N_GPU][NUM_LAYER];
Parameter *ln_2_b[N_GPU][NUM_LAYER], *ln_2_g[N_GPU][NUM_LAYER];
Parameter *mlp1_b[N_GPU][NUM_LAYER], *mlp1_w[N_GPU][NUM_LAYER];
Parameter *mlp2_b[N_GPU][NUM_LAYER], *mlp2_w[N_GPU][NUM_LAYER];
Parameter *ln_f_b[N_GPU], *ln_f_g[N_GPU];
Parameter *wpe[N_GPU], *wte[N_GPU];


void alloc_and_set_parameters(float *param) {
  size_t pos = 0;
  int order[] = {
      0, 1, 10, 11, 2, 3, 4, 5, 6, 7, 8, 9,
  };

  // debuging
  // int deviceCount = 0;
  // cudaGetDeviceCount(&deviceCount);
  // printf("# of gpus : %d \n", deviceCount);

  /* set model parmeters to each GPU */
  for (int idx = 0; idx < N_GPU; idx++) {

    CHECK_CUDA(cudaSetDevice(idx));
    pos = 0; // initialize position

    for (int i = 0; i < NUM_LAYER; i++) {
      attn_b[idx][order[i]] = new Parameter({3 * HIDDEN_DIM}, param + pos);
      pos += OFFSET1;
      attn_w[idx][order[i]] = new Parameter({HIDDEN_DIM, 3 * HIDDEN_DIM}, param + pos);
      pos += OFFSET2;
      proj_b[idx][order[i]] = new Parameter({HIDDEN_DIM}, param + pos);
      pos += OFFSET3;
      proj_w[idx][order[i]] = new Parameter({HIDDEN_DIM, HIDDEN_DIM}, param + pos);
      pos += OFFSET4;
      ln_1_b[idx][order[i]] = new Parameter({HIDDEN_DIM}, param + pos);
      pos += OFFSET3;
      ln_1_g[idx][order[i]] = new Parameter({HIDDEN_DIM}, param + pos);
      pos += OFFSET3;
      ln_2_b[idx][order[i]] = new Parameter({HIDDEN_DIM}, param + pos);
      pos += OFFSET3;
      ln_2_g[idx][order[i]] = new Parameter({HIDDEN_DIM}, param + pos);
      pos += OFFSET3;
      mlp1_b[idx][order[i]] = new Parameter({4 * HIDDEN_DIM}, param + pos);
      pos += OFFSET5;
      mlp1_w[idx][order[i]] = new Parameter({HIDDEN_DIM, 4 * HIDDEN_DIM}, param + pos);
      pos += OFFSET6;
      mlp2_b[idx][order[i]] = new Parameter({HIDDEN_DIM}, param + pos);
      pos += OFFSET3;
      mlp2_w[idx][order[i]] = new Parameter({4 * HIDDEN_DIM, HIDDEN_DIM}, param + pos);
      pos += OFFSET6;
    }
    ln_f_b[idx] = new Parameter({HIDDEN_DIM}, param + pos);
    pos += OFFSET3;
    ln_f_g[idx] = new Parameter({HIDDEN_DIM}, param + pos);
    pos += OFFSET3;
    wpe[idx] = new Parameter({MAX_SEQ_LEN, HIDDEN_DIM}, param + pos);
    pos += OFFSET7;
    wte[idx] = new Parameter({NUM_VOCAB, HIDDEN_DIM}, param + pos);
    pos += OFFSET8;

    // alloc activations for each GPU
    alloc_activations_batch_multi_gpu(tokens_per_prompt + num_generated_tokens, BATCH_SIZE, idx);
  }
}

void free_parameters() {

  for (int idx = 0; idx < N_GPU; idx++) {
    CHECK_CUDA(cudaSetDevice(idx));
    for (int i = 0; i < NUM_LAYER; i++) {
      delete attn_b[idx][i];
      delete attn_w[idx][i];
      delete proj_b[idx][i];
      delete proj_w[idx][i];
      delete ln_1_b[idx][i];
      delete ln_1_g[idx][i];
      delete ln_2_b[idx][i];
      delete ln_2_g[idx][i];
      delete mlp1_b[idx][i];
      delete mlp1_w[idx][i];
      delete mlp2_b[idx][i];
      delete mlp2_w[idx][i];
    }
    delete ln_f_b[idx];
    delete ln_f_g[idx];
    delete wpe[idx];
    delete wte[idx];

    // free activations for each GPU
    free_activations_multi_gpu(idx);
  }
}


/////////////////////////////////////////////////////////// multi-GPU version layers ///////////////////////////////////////////////////////////
void ffn_batch_multi_gpu(Activation *in, Parameter *mlp1_w, Parameter *mlp1_b,
         Parameter *mlp2_w, Parameter *mlp2_b, Activation *out, size_t idx, size_t token_num) {
  /* Projection Up:
    [b, seq_len, HIDDEN_DIM] -> [b, seq_len, 4*HIDDEN_DIM] */
  linear_batch_cuda(in, mlp1_w, mlp1_b, ffn_proj_a[idx], token_num);

  /* GELU */
  gelu_batch_cuda(ffn_proj_a[idx]);

  /* Projection Down:
    [b, seq_len, 4*HIDDEN_DIM] -> [b, seq_len, HIDDEN_DIM] */
  linear_batch_cuda(ffn_proj_a[idx], mlp2_w, mlp2_b, out, token_num);
}


/* Attention_batch
 * @param [in1]    q: [b, seq_len, HIDDEN_DIM/NUM_HEAD]
 * @param [in2]    k: [b, seq_len, HIDDEN_DIM/NUM_HEAD]
 * @param [in3]    v: [b, seq_len, HIDDEN_DIM/NUM_HEAD]
 * @param [in4] mask: [b, seq_len, seq_len]
 * @param [out]  out: [b, seq_len, HIDDEN_DIM/NUM_HEAD]
 */
void attention_batch_multi_gpu(Activation *q, Activation *k, Activation *v, Activation *mask, 
               Activation *out, size_t idx, size_t token_num) {
  /* Get Attention score by q @ k */
  transpose_batch_cuda(k, k_transposed_a[idx]);
  matmul_batch_cuda(q, k_transposed_a[idx], attn_score_a[idx], token_num);

  /* Scaling */
  scaling_batch_cuda(attn_score_a[idx], (1.0 / sqrt(k->shape[2])));

  /* Masking */
  add_batch_cuda(attn_score_a[idx], mask);

  /* Softmax */
  softmax_batch_cuda(attn_score_a[idx]);

  /* Attention score @ v */
  matmul_batch_cuda(attn_score_a[idx], v, out, token_num);
}


/* (Masked) Multi-Head Self Attention batch
 * @param [in1]     in: [b, seq_len, HIDDEN_DIM]
 * @param [in2] attn_b: [3*HIDDEN_DIM]
 * @param [in3] attn_w: [HIDDEN_DIM, 3*HIDDEN_DIM]
 * @param [in4] proj_b: [HIDDEN_DIM]
 * @param [in5] proj_w: [HIDDEN_DIM, HIDDEN_DIM]
 * @param [out]    out: [b, seq_len, HIDDEN_DIM]
 */
void mha_batch_multi_gpu(Activation *in, Parameter *attn_b, Parameter *attn_w,
         Parameter *proj_b, Parameter *proj_w, Activation *out, size_t idx, size_t layer_idx, size_t token_num) {
  /* QKV projection:
    [b, seq_len, HIDDEN_DIM] ->
    [b, seq_len, 3*HIDDEN_DIM] */
  linear_batch_cuda(in, attn_w, attn_b, mha_qkv_proj_a[idx], token_num);

  /* Split into Q, K, V:
    [b, seq_len, 3*HIDDEN_DIM] ->
    [b, 3, seq_len, HIDDEN_DIM] */
  split_qkv_batch_cuda(mha_qkv_proj_a[idx], mha_split_qkv_a[idx]);

  /* Split into multiple heads:
    [b, 3, seq_len, HIDDEN_DIM] ->
    [b, 3, NUM_HEAD, seq_len, HIDDEN_DIM/NUM_HEAD] */
  split_head_batch_cuda(mha_split_qkv_a[idx], NUM_HEAD, mha_split_head_a[idx]);

  /* Generate mask to hide future inputs */
  generate_mask_batch_cuda(mha_mask_a[idx], token_num); 

  /* Perform Attention over each head:
    [NUM_HEAD, 3, seq_len, HIDDEN_DIM/NUM_HEAD] ->
    [NUM_HEAD, seq_len, HIDDEN_DIM/NUM_HEAD] */
  for (size_t i = 0; i < NUM_HEAD; i++) {
    /* Extract Q, K, V from qkv_head */
    // extract_qkv_batch_cuda(mha_split_head_a[idx], i, NUM_HEAD, mha_q_a[idx], mha_k_a[idx], mha_v_a[idx]);
    extract_qkv_batch_cuda(mha_split_head_a[idx], i, NUM_HEAD, mha_q_a[idx], k_temp_a[idx], v_temp_a[idx]); // should be changed to this
    store_kv_cache(k_cache_a[idx], v_cache_a[idx], k_temp_a[idx], v_temp_a[idx], layer_idx, i, token_num);
    insert_kv_cache(k_cache_a[idx], v_cache_a[idx], mha_k_a[idx], mha_v_a[idx], layer_idx, i, token_num);

    /* Attention */
    attention_batch_multi_gpu(mha_q_a[idx], mha_k_a[idx], mha_v_a[idx], mha_mask_a[idx], mha_attn_out_a[idx], idx, token_num);

    /* Merge each head's attn output
      [b, seq_len, HIDDEN_DIM/NUM_HEAD] ->
      [b, NUM_HEAD, seq_len, HIDDEN_DIM/NUM_HEAD] */
    merge_head_batch_cuda(mha_attn_out_a[idx], i, NUM_HEAD, mha_merge_head_a[idx]);
  }

  /* Concat each heads:
    [b, NUM_HEAD, seq_len, HIDDEN_DIM/NUM_HEAD] ->
    [b, seq_len, HIDDEN_DIM] */
  concat_head_batch_cuda(mha_merge_head_a[idx], mha_concat_head_a[idx]);

  /* OUT projection:
    [b, seq_len, HIDDEN_DIM] -> [b, seq_len, HIDDEN_DIM] */
  linear_batch_cuda(mha_concat_head_a[idx], proj_w, proj_b, out, token_num);
}


/* Transformer Block batch
 * @param [in1]      in: [b, seq_len, HIDDEN_DIM]
 * @param [in2]  attn_b: [3*HIDDEN_DIM]
 * @param [in3]  attn_w: [HIDDEN_DIM, 3*HIDDEN_DIM]
 * @param [in4]  proj_b: [HIDDEN_DIM]
 * @param [in5]  proj_w: [HIDDEN_DIM, HIDDEN_DIM]
 * @param [in6]  ln_1_b: [HIDDEN_DIM]
 * @param [in7]  ln_1_g: [HIDDEN_DIM]
 * @param [in8]  ln_2_b: [HIDDEN_DIM]
 * @param [in9]  ln_2_g: [HIDDEN_DIM]
 * @param [in10] mlp1_b: [4*HIDDEN_DIM]
 * @param [in11] mlp1_w: [HIDDEN_DIM, 4*HIDDEN_DIM]
 * @param [in12] mlp2_b: [HIDDEN_DIM]
 * @param [in13] mlp2_w: [4*HIDDEN_DIM, HIDDEN_DIM]
 * @param [out]     out: [b, seq_len, HIDDEN_DIM]
 */
void transformer_block_batch_multi_gpu(Activation *in, Parameter *attn_b, Parameter *attn_w,
                       Parameter *proj_b, Parameter *proj_w, Parameter *ln_1_b,
                       Parameter *ln_1_g, Parameter *ln_2_b, Parameter *ln_2_g,
                       Parameter *mlp1_b, Parameter *mlp1_w, Parameter *mlp2_b,
                       Parameter *mlp2_w, Activation *out, size_t idx, size_t layer_idx, size_t token_num) {
  /* Copy Residual */
  copy_batch_cuda(in, residual_a[idx]);

  /* Layer Normalization */
  layer_norm_batch_cuda(in, ln_1_g, ln_1_b);

  /* Masked Multi-Head Self-Attention */
  mha_batch_multi_gpu(in, attn_b, attn_w, proj_b, proj_w, mha_out_a[idx], idx, layer_idx, token_num);

  /* Add Residual */
  add_batch_cuda(mha_out_a[idx], residual_a[idx]);

  /* Copy Residual */
  copy_batch_cuda(mha_out_a[idx], residual_a[idx]);

  /* Layer Normalization */
  layer_norm_batch_cuda(mha_out_a[idx], ln_2_g, ln_2_b);

  /* Position-wise Feed-Forward Network */
  ffn_batch_multi_gpu(mha_out_a[idx], mlp1_w, mlp1_b, mlp2_w, mlp2_b, out, idx, token_num);

  /* Add Residual */
  add_batch_cuda(out, residual_a[idx]);
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


/* multi-GPU version */
void generate_tokens(int *input, int *output, size_t n_prompt, size_t n_token) {
  int mpi_rank, mpi_world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);

  int n_prompt_per_node = n_prompt / mpi_world_size;
  assert(n_prompt % mpi_world_size == 0); // assume equal workload partitioning

  /* allocate memory for input, output (nodes for rank != 0) */
  if (mpi_rank != 0) {
    input = (int *)malloc(n_prompt_per_node * tokens_per_prompt * sizeof(int));
    output = (int *)malloc(n_prompt_per_node * n_token * sizeof(int));
  }

  /* MPI Scatter */
  if (mpi_world_size > 1) {
    MPI_Scatter(input, n_prompt_per_node * tokens_per_prompt, MPI_INT, input, n_prompt_per_node * tokens_per_prompt, MPI_INT, 0, MPI_COMM_WORLD);
  }

  const size_t n_prompt_per_gpu = n_prompt_per_node / N_GPU;
  assert(n_prompt_per_gpu % BATCH_SIZE == 0); // for compact batching
  size_t n_batches = n_prompt_per_gpu / BATCH_SIZE;

  // main logic
  #pragma omp parallel for num_threads(N_GPU)
  for (int idx = 0; idx < N_GPU; idx++) {
    CHECK_CUDA(cudaSetDevice(idx));

    for (size_t b = 0; b < n_batches; b++) {
      int batched_prompt_size = tokens_per_prompt * BATCH_SIZE;
      int prompt_size = tokens_per_prompt;

      /* initialize prompt batch */
      vector<int> batched_input_prompt[BATCH_SIZE];
      for (size_t i = 0; i < BATCH_SIZE; i++) {
        batched_input_prompt[i].resize(prompt_size);
        memcpy(batched_input_prompt[i].data(), input + idx * (n_prompt_per_gpu * tokens_per_prompt) + b * batched_prompt_size + i * prompt_size,
              prompt_size * sizeof(int));
      }

      /* Inner loop : generate next token */
      for (size_t t = 0; t < n_token; t++) {
        /* Initialize activations */
        resize_activations_batch_multi_gpu(prompt_size, BATCH_SIZE, idx, t); // resize at each token pos generation

        /* Token + Positional Embedding */
        token_pos_embedding_batch_cuda(batched_input_prompt, wte[idx], wpe[idx], embd_a[idx], t);

        /* Forward path of Transformer blocks */
        for (size_t l = 0; l < NUM_LAYER; l++) {
          transformer_block_batch_multi_gpu(embd_a[idx], attn_b[idx][l], attn_w[idx][l], proj_b[idx][l], proj_w[idx][l],
                            ln_1_b[idx][l], ln_1_g[idx][l], ln_2_b[idx][l], ln_2_g[idx][l],
                            mlp1_b[idx][l], mlp1_w[idx][l], mlp2_b[idx][l], mlp2_w[idx][l],
                            transformer_block_a[idx], idx, l, t);

          /* Copy output to embd_a for next block */
          copy_batch_cuda(transformer_block_a[idx], embd_a[idx]);
        }

        /* Final Layer Normalization */
        layer_norm_batch_cuda(embd_a[idx], ln_f_g[idx], ln_f_b[idx]);

        /* Projection to vocab. dimension */
        transpose_cuda(wte[idx], wte_transposed_a[idx]);

        /* select only the last token from embd_a[idx] */
        select_last_token_cuda(embd_a[idx], embd_a_last[idx]);

        // changed embd_a[idx] -> embd_a_last[idx] in matmul_batch_final_cuda
        matmul_batch_final_cuda(embd_a_last[idx], wte_transposed_a[idx], logit_a[idx], t);

        /* greedy sampling */
        vector<int> next_token_id = top1_sampling_batch(logit_a[idx]);
        for (size_t i = 0; i < BATCH_SIZE; i++) {
          batched_input_prompt[i].push_back(next_token_id[i]);
        }
        prompt_size += 1;

        /* store generated token to output */
        for (size_t i = 0; i < BATCH_SIZE; i++) {
          output[idx * (n_token * n_prompt_per_gpu) + b * (n_token * BATCH_SIZE) + i * n_token + t] = next_token_id[i];
        }
      }
    }
  }

  /* MPI Gather */ 
  if (mpi_world_size > 1) {
    MPI_Gather(output, n_prompt_per_node * n_token, MPI_INT, output, n_prompt_per_node * n_token, MPI_INT, 0, MPI_COMM_WORLD);
  }

  /* free allocated memory for input, output */
  if (mpi_rank != 0) {
    free(input);
    free(output);
  }

}
