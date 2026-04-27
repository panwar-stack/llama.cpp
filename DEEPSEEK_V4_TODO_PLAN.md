# DeepSeek-V4 Known Limitations: Implementation Plan

## Overview

This document outlines a phased plan for implementing the 5 known limitations left as TODOs in the initial DeepSeek-V4-Flash support. The plan prioritizes items by impact, feasibility, and dependency order.

| # | Limitation | Priority | Complexity | Impact on Quality/Efficiency |
|---|-----------|----------|------------|------------------------------|
| 1 | Hyper-Connections (HC) | High | High | High — fundamental to V4's architecture |
| 2 | KV Compression + Indexer | High | Very High | High — enables long-context efficiency |
| 3 | Hash-based Routing | Medium | Medium | Low-Medium — only first 3 layers |
| 4 | Custom Sparse Attention Kernel | Medium | High | High — performance optimization for long context |
| 5 | MTP (Multi-Token Prediction) | Low | High | Medium — inference speedup feature |

---

## Phase 1: Hyper-Connections (HC)

### Why First
Hyper-Connections replace the standard residual pathway (`x = x + attn_out`). Without them, the model uses plain residuals, which deviates from V4's training architecture and may cause measurable quality degradation.

### Architecture Recap (from config & inference code)
- `hc_mult = 4`: number of parallel state copies
- `hc_eps = 1e-6`: small constant for numerical stability
- `hc_sinkhorn_iters = 20`: Sinkhorn normalization iterations
- Tensors per layer:
  - `hc_attn_fn` / `hc_ffn_fn`: mixing weights
  - `hc_attn_base` / `hc_ffn_base`: bias terms
  - `hc_attn_scale` / `hc_ffn_scale`: scale factors (length-3)

### Implementation Steps

#### 1.1 Add HC Tensors to GGUF Schema
**Files:** `gguf-py/gguf/constants.py`, `src/llama-arch.h/cpp`, `gguf-py/gguf/tensor_mapping.py`

Add new `MODEL_TENSOR` entries:
```python
HC_ATTN_FN    = auto()
HC_FFN_FN     = auto()
HC_ATTN_BASE  = auto()
HC_FFN_BASE   = auto()
HC_ATTN_SCALE = auto()
HC_FFN_SCALE  = auto()
HC_PRE        = auto()  # optional, if HF uses pre/post proj matrices
HC_POST       = auto()
```

Map HF names in `tensor_mapping.py`:
```python
"model.layers.{bid}.hc_attn_fn",
"model.layers.{bid}.hc_ffn_fn",
# ... etc
```

#### 1.2 Add HC Tensors to C++ Layer Struct
**File:** `src/llama-model.h`

```cpp
struct llama_layer {
    // ... existing tensors ...
    struct ggml_tensor * hc_attn_fn    = nullptr;
    struct ggml_tensor * hc_ffn_fn     = nullptr;
    struct ggml_tensor * hc_attn_base  = nullptr;
    struct ggml_tensor * hc_ffn_base   = nullptr;
    struct ggml_tensor * hc_attn_scale = nullptr;
    struct ggml_tensor * hc_ffn_scale  = nullptr;
};
```

#### 1.3 Load HC Tensors in Model Loader
**File:** `src/llama-model.cpp`

In `LLM_ARCH_DEEPSEEK4` tensor loading block:
```cpp
layer.hc_attn_fn    = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN,    "weight", i), {(2 + hc_mult) * hc_mult, hc_mult * n_embd}, 0);
layer.hc_ffn_fn     = create_tensor(tn(LLM_TENSOR_HC_FFN_FN,     "weight", i), {(2 + hc_mult) * hc_mult, hc_mult * n_embd}, 0);
layer.hc_attn_base  = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE,  "weight", i), {(2 + hc_mult) * hc_mult}, 0);
layer.hc_ffn_base   = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE,   "weight", i), {(2 + hc_mult) * hc_mult}, 0);
layer.hc_attn_scale = create_tensor(tn(LLM_TENSOR_HC_ATTN_SCALE, "weight", i), {3}, 0);
layer.hc_ffn_scale  = create_tensor(tn(LLM_TENSOR_HC_FFN_SCALE,  "weight", i), {3}, 0);
```

#### 1.4 Implement HC in Graph Builder
**File:** `src/models/deepseek4.cpp`

Replace the current residual path with HC logic:

```cpp
// Before the attention layer, expand state to hc_mult copies
// state shape: [n_embd, hc_mult, n_tokens]
ggml_tensor * state = ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens);
state = ggml_repeat(ctx0, state, ggml_new_tensor_3d(ctx0, inpL->type, n_embd, hc_mult, n_tokens));

// --- HC Pre-Mix for Attention ---
// Flatten state: [hc_mult * n_embd, n_tokens]
ggml_tensor * state_flat = ggml_reshape_2d(ctx0, state, hc_mult * n_embd, n_tokens);

// Compute mixes = sigmoid(fn * state + base) * scale + eps
ggml_tensor * mixes = ggml_mul_mat(ctx0, model.layers[il].hc_attn_fn, state_flat);
mixes = ggml_add(ctx0, mixes, model.layers[il].hc_attn_base);
mixes = ggml_sigmoid(ctx0, mixes);
// Apply scale[0], scale[1], scale[2] to respective sections

// Split mixes into pre, post, comb
// pre = mixes[:, :hc_mult]  -> shape [hc_mult, n_tokens]
// post = mixes[:, hc_mult:2*hc_mult]
// comb = mixes[:, 2*hc_mult:] -> reshape to [hc_mult, hc_mult, n_tokens]

// Sinkhorn normalization on comb (iterative row/col softmax)
// For 20 iterations, each iteration is:
//   comb = softmax(comb, dim=1)  // row-wise
//   comb = softmax(comb, dim=0)  // col-wise
// In GGML, use ggml_soft_max_ext with appropriate masks.
// This creates a large static graph (~160 ops per layer). Acceptable for MVP.

// Apply pre-mix: reduce hc_mult copies to single vector
ggml_tensor * pre = ggml_view_2d(ctx0, mixes, hc_mult, n_tokens, mixes->nb[1], 0);
ggml_tensor * attn_inp = ggml_sum_rows(ctx0, ggml_mul(ctx0, state_flat, pre));
attn_inp = ggml_reshape_2d(ctx0, attn_inp, n_embd, n_tokens);

// --- Run Attention + FFN on attn_inp (existing logic) ---
// ... attention and FFN code remains the same, operating on attn_inp ...

// --- HC Post-Mix ---
// Apply post and comb to distribute output back to hc_mult state copies
ggml_tensor * post = ggml_view_2d(ctx0, mixes, hc_mult, n_tokens, mixes->nb[1], hc_mult * mixes->nb[0]);
// new_state = post * ffn_out + comb @ state
// ... reshape and combine ...

// The final output for the next layer is the sum or mean of the hc_mult copies
```

**Performance Note:** 20 iterations of Sinkhorn per layer × 43 layers = 860 extra ops. This may slow graph compilation. Consider adding a custom `ggml_sinkhorn` fused op in a follow-up PR.

**Simplification for initial PR:** Implement 5 iterations instead of 20, with a comment noting the full 20 are needed for parity. Or implement the composite ops and measure performance.

---

## Phase 2: KV Compression + Indexer

### Why Second
Without KV compression, V4 cannot achieve its 1M context length efficiently. However, implementing compression requires understanding how V4 manages its hybrid cache (window + compressed).

### Architecture Recap
- `sliding_window = 128`: recent tokens kept in a ring buffer with standard RoPE (`rope_theta=10000`)
- `compress_ratios`: per-layer array. Values are `0` (no compression), `4` (heavy compression + indexer), `128` (light compression)
- `compress_rope_theta = 160000`: different RoPE base for compressed positions
- For `compress_ratio == 4` layers, an **Indexer** selects top-512 compressed positions

### Implementation Steps

#### 2.1 Design Decision: Custom Memory Context
**Do NOT modify `llama_kv_cache.cpp`**. V4's cache layout is too different. Instead:

Add a new `llama_memory_v4_context` class (similar to `llama_memory_recurrent_context` for Mamba/RWKV):

**Files:** New `src/llama-memory-v4.h` / `src/llama-memory-v4.cpp`

```cpp
struct llama_memory_v4_context : llama_memory_context_i {
    // Per-layer buffers
    struct layer_cache {
        // Window part: ring buffer of sliding_window tokens
        ggml_tensor * k_window = nullptr;  // [head_dim, sliding_window]
        ggml_tensor * v_window = nullptr;  // [head_dim, sliding_window]
        int window_pos = 0;  // write position in ring

        // Compressed part: linear buffer
        ggml_tensor * k_compress = nullptr;  // [head_dim, max_seq / ratio]
        ggml_tensor * v_compress = nullptr;  // [head_dim, max_seq / ratio]
        int compress_count = 0;

        // For ratio==4: indexer top-k indices
        ggml_tensor * topk_indices = nullptr;  // [indexer_top_k]
    };

    std::vector<layer_cache> layers;
};
```

#### 2.2 Add Compressor/Indexer Tensors
**Files:** `gguf-py/gguf/constants.py`, `src/llama-arch.h/cpp`, `src/llama-model.h/cpp`, `gguf-py/gguf/tensor_mapping.py`

New tensor types:
```cpp
LLM_TENSOR_COMPRESS_WKV,    // compressor KV projection
LLM_TENSOR_COMPRESS_WGATE,  // compressor gate
LLM_TENSOR_COMPRESS_APE,    // absolute positional embedding for compressed positions
LLM_TENSOR_COMPRESS_NORM,   // RMS norm after compression
LLM_TENSOR_INDEXER_PROJ,    // indexer query projection
LLM_TENSOR_INDEXER_ATTN_K,  // indexer key projection
LLM_TENSOR_INDEXER_ATTN_Q_B,// indexer query bias
```

#### 2.3 Implement Compression Subgraph
**File:** `src/models/deepseek4.cpp`

In each layer, after computing KV but before storing to cache:

```cpp
if (hparams.compress_ratios[il] > 0) {
    uint32_t ratio = hparams.compress_ratios[il];

    // Project and gate the KV state
    ggml_tensor * k_comp = ggml_mul_mat(ctx0, model.layers[il].compress_wkv, kv);
    ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_mul_mat(ctx0, model.layers[il].compress_wgate, kv));
    k_comp = ggml_mul(ctx0, k_comp, gate);

    // Add compressed positional embedding
    // The positions for compressed tokens are downsampled
    ggml_tensor * comp_pos = build_inp_pos_compress();  // new input for compressed positions
    k_comp = ggml_add(ctx0, k_comp, model.layers[il].compress_ape);

    // Downsample sequence dimension by 'ratio'
    // This requires reshaping [head_dim, n_tokens] -> [head_dim * ratio, n_tokens / ratio]
    // then taking the mean over each ratio-sized chunk
    // In GGML: reshape to [head_dim, ratio, n_tokens/ratio], then mean over dim 1
    k_comp = ggml_reshape_3d(ctx0, k_comp, head_dim, ratio, n_tokens / ratio);
    k_comp = ggml_mean(ctx0, k_comp, 1);  // mean over ratio dimension
    // Note: ggml_mean may not exist; alternative: sum + scale

    k_comp = build_norm(k_comp, model.layers[il].compress_norm, nullptr, LLM_NORM_RMS, il);

    // Store k_comp to persistent compressed KV buffer via ggml_set_rows
    // ... (requires custom memory context integration)
}
```

#### 2.4 Integrate with Attention
Modify the attention call to read from both window and compressed buffers:

```cpp
// Gather KV from window (recent 128 tokens)
ggml_tensor * k_window = mctx->get_window_k(il);
ggml_tensor * v_window = mctx->get_window_v(il);

// Gather KV from compressed buffer
ggml_tensor * k_compress = mctx->get_compress_k(il);
ggml_tensor * v_compress = mctx->get_compress_v(il);

// For ratio==4, use indexer to select top-k from compressed
ggml_tensor * k_compress_topk = k_compress;
ggml_tensor * v_compress_topk = v_compress;
if (ratio == 4) {
    // Run indexer attention to select top-512 positions
    // Query = current hidden state projected via indexer_attn_q_b
    // Key = compressed KV projected via indexer_attn_k
    // Score = softmax(Q @ K^T), topk = argmax(score, k=512)
    ggml_tensor * idx_q = ggml_mul_mat(ctx0, model.layers[il].indexer_attn_q_b, cur);
    ggml_tensor * idx_k = ggml_mul_mat(ctx0, model.layers[il].indexer_attn_k, k_compress);
    ggml_tensor * idx_score = ggml_soft_max(ctx0, ggml_mul_mat(ctx0, idx_k, idx_q));
    ggml_tensor * topk_idx = ggml_top_k(ctx0, idx_score, hparams.indexer_top_k);

    k_compress_topk = ggml_get_rows(ctx0, k_compress, topk_idx);
    v_compress_topk = ggml_get_rows(ctx0, v_compress, topk_idx);
}

// Concatenate window and compressed (topk) KV
ggml_tensor * K_all = ggml_concat(ctx0, k_window, k_compress_topk, 1);  // along sequence dim
ggml_tensor * V_all = ggml_concat(ctx0, v_window, v_compress_topk, 1);

// Run attention over gathered K_all, V_all
cur = build_attn(..., Qcur, K_all, V_all, ..., kq_scale, il);
```

**Note:** `ggml_top_k` may not exist. If so, implement indexer-less compression first (attend to all compressed tokens), which is still a major memory win.

#### 2.5 Hook into llama_context
**File:** `src/llama-context.cpp`

The `llama_context` needs to instantiate `llama_memory_v4_context` instead of `llama_kv_cache` when `arch == LLM_ARCH_DEEPSEEK4`.

---

## Phase 3: Hash-based Routing

### Why Third
Only affects the first 3 layers. The fallback to dense FFN is acceptable for many use cases.

### Implementation

#### 3.1 Pass Token IDs Through Graph
**File:** `src/llama-graph.cpp`

Add a new graph input for raw token IDs (similar to `inp_pos`):
```cpp
ggml_tensor * build_inp_token_ids() const;
```

Store `token_ids` in `llm_graph_input_kv` or create a separate input struct.

#### 3.2 Add tid2eid Tensor
**File:** `src/llama-model.cpp`

Load the hash lookup table:
```cpp
layer.tid2eid = create_tensor(tn(LLM_TENSOR_TID2EID, "weight", i), {n_vocab, n_expert_used}, 0);
```

#### 3.3 Use Hash Routing in MoE
**File:** `src/models/deepseek4.cpp`

For hash layers (`il < n_hash_layers`):
```cpp
if (il < (int) hparams.n_hash_layers && model.layers[il].tid2eid) {
    // Look up expert indices from token IDs
    ggml_tensor * indices = ggml_get_rows(ctx0, model.layers[il].tid2eid, inp_token_ids);
    // indices shape: [n_expert_used, n_tokens]
    // Use indices directly for expert selection instead of computing logits
    cur = build_moe_ffn_hash(cur, indices, model.layers[il].ffn_up_exps, ...);
} else {
    // Standard score-based routing
    cur = build_moe_ffn(cur, model.layers[il].ffn_gate_inp, ...);
}
```

**Note:** `build_moe_ffn_hash` would be a variant of `build_moe_ffn` that takes pre-computed expert indices instead of computing them from logits. This requires adding a new overload or parameter to the existing function.

---

## Phase 4: Custom Sparse Attention Kernel

### Why Fourth
This is a performance optimization, not a correctness requirement. The gather-based approach (Phase 2) works but is slower than a fused kernel.

### Implementation Options

#### Option A: Fused Gather + Attention (Recommended for initial PR)
Use `ggml_get_rows` to gather KV by topk indices, then run standard `ggml_flash_attn_ext`. This is already described in Phase 2.

#### Option B: New GGML Op `ggml_sparse_attn_ext`
**File:** New op in `ggml/src/ggml.c` + backend kernels

Add a new operation that:
1. Takes Q, K, V, and a sparse index tensor
2. Computes attention only over the indexed positions
3. Returns the attended output

This is a large project requiring CPU, CUDA, and Metal kernels. It should be done in a separate PR focused purely on the GGML op.

---

## Phase 5: MTP (Multi-Token Prediction)

### Why Last
MTP is an inference optimization (speculative decoding) that doesn't affect correctness. The model works fine without it.

### Implementation
MTP adds 1 or more auxiliary decoder heads that predict future tokens. To support it:
1. In the converter, stop skipping MTP layers (`skip_mtp = False`)
2. Load MTP weights (they are already in the HF checkpoint)
3. Add MTP-specific tensors: `nextn_eh_proj`, `nextn_enorm`, etc. (some already exist in the codebase)
4. In the graph builder, add MTP heads after the main transformer
5. Integrate with llama.cpp's speculative decoding path

This is complex because it requires modifying the sampling loop, not just the graph builder.

---

## Suggested Development Order

For a contributor wanting to tackle these incrementally:

1. **Start with HC (Phase 1)** — it affects every layer and has the biggest quality impact. Begin with a simplified Sinkhorn (e.g., 5 iterations) and iterate.
2. **Add Hash Routing (Phase 3)** — only 3 layers, relatively isolated. Good "warm-up" task before tackling compression.
3. **Implement KV Compression without Indexer (Phase 2 lite)** — add the compressor, pool by ratio, store to persistent buffers, attend to all compressed positions. Skip the indexer/top-k logic initially.
4. **Add Indexer + Sparse Attention (Phase 2 full + Phase 4)** — add top-k selection via indexer, then optimize with a custom gather kernel.
5. **MTP (Phase 5)** — lowest priority, do last.

---

## Files to Create/Modify (Complete List)

### New Files
- `src/models/deepseek4.cpp` — already exists, will be heavily modified for HC and compression
- `src/llama-memory-v4.h` — new memory backend for V4's hybrid cache
- `src/llama-memory-v4.cpp`

### Modified Files (by phase)

**Phase 1 (HC):**
- `gguf-py/gguf/constants.py` — new MODEL_TENSOR entries
- `gguf-py/gguf/tensor_mapping.py` — HC tensor name mappings
- `convert_hf_to_gguf.py` — write HC metadata
- `src/llama-arch.h` — new LLM_TENSOR enum values
- `src/llama-arch.cpp` — tensor name and op info mappings
- `src/llama-model.h` — HC tensor pointers in llama_layer
- `src/llama-model.cpp` — HC tensor loading
- `src/models/deepseek4.cpp` — HC graph logic

**Phase 2 (Compression):**
- `gguf-py/gguf/constants.py` — COMPRESS_*, INDEXER_* tensors
- `gguf-py/gguf/tensor_mapping.py` — compressor/indexer mappings
- `convert_hf_to_gguf.py` — write compression metadata
- `src/llama-arch.h/cpp` — new tensor enums and names
- `src/llama-model.h` — compressor/indexer tensor pointers
- `src/llama-model.cpp` — compressor/indexer tensor loading
- `src/llama-context.cpp` — instantiate V4 memory backend
- `src/models/deepseek4.cpp` — compression subgraph + hybrid attention

**Phase 3 (Hash Routing):**
- `src/llama-graph.h/cpp` — `build_inp_token_ids()` helper
- `src/llama-model.h` — `tid2eid` pointer
- `src/llama-model.cpp` — `tid2eid` loading
- `src/llama-graph.cpp` — `build_moe_ffn_hash()` variant
- `src/models/deepseek4.cpp` — hash routing branch

**Phase 4 (Sparse Kernel):**
- `ggml/src/ggml.h` — `ggml_sparse_attn_ext` declaration
- `ggml/src/ggml.c` — op definition
- Backend-specific implementations (CPU, CUDA, Metal)

**Phase 5 (MTP):**
- `convert_hf_to_gguf.py` — `skip_mtp = False`
- `src/models/deepseek4.cpp` — MTP graph heads
- Sampling/decoding loop changes
