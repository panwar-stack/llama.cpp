# DeepSeek V4 Support Implementation Plan for llama.cpp

## Executive Summary

DeepSeek V4 introduces several architectural novelties on top of the V2/V3 foundation: **Hyper-Connections (HC)** replacing residuals, **per-layer KV compression** with a learned compressor/indexer, **low-rank output projection**, **hash-based expert routing** for early layers, and `sqrtsoftplus` gating. The existing `DEEPSEEK2` architecture in llama.cpp already covers MLA (low-rank Q, KV lora), YaRN RoPE, MoE with shared experts, and `e_score_correction_bias`. This plan details how to incrementally add V4 support by introducing a new `LLM_ARCH_DEEPSEEK4`, extending the GGUF schema, converter, and graph builder, while maximizing reuse of existing code paths.

---

## Phase 1: GGUF Schema & Python Converter

### 1.1 Add Architecture Constant

**Files:**
- `gguf-py/gguf/constants.py`
- `src/llama-arch.h`
- `src/llama-arch.cpp`

**Changes:**
1. In `gguf-py/gguf/constants.py` (`MODEL_ARCH` enum), add:
   ```python
   DEEPSEEK4 = auto()
   ```
2. In `MODEL_ARCH_NAMES` mapping:
   ```python
   MODEL_ARCH.DEEPSEEK4: "deepseek4",
   ```
3. In `src/llama-arch.h` (`llm_arch` enum), add:
   ```cpp
   LLM_ARCH_DEEPSEEK4,
   ```
   (Insert before `LLM_ARCH_UNKNOWN`.)
4. In `src/llama-arch.cpp` (`LLM_ARCH_NAMES`):
   ```cpp
   { LLM_ARCH_DEEPSEEK4, "deepseek4" },
   ```

### 1.2 Add New KV Metadata Keys

**File:** `src/llama-arch.h`

Add the following to `llm_kv` (and corresponding strings in `llama-arch.cpp`):

| C++ Enum | GGUF Key Name | Source Config Field |
|----------|---------------|---------------------|
| `LLM_KV_HYPER_CONNECTION_MULT` | `%s.hyper_connection.mult` | `hc_mult` |
| `LLM_KV_HYPER_CONNECTION_EPS` | `%s.hyper_connection.eps` | `hc_eps` |
| `LLM_KV_ATTENTION_O_LORA_RANK` | `%s.attention.o_lora_rank` | `o_lora_rank` |
| `LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT` | `%s.attention.output_group_count` | `o_groups` |
| `LLM_KV_HASH_LAYER_COUNT` | `%s.hash_layer_count` | `num_hash_layers` |
| `LLM_KV_ROPE_FREQ_BASE_COMPRESS` | `%s.rope.freq_base_compress` | `compress_rope_theta` |
| `LLM_KV_ATTENTION_COMPRESS_RATIO` | `%s.attention.layer_compress_ratio` | `compress_ratios` (per-layer array) |

**Note:** `LLM_KV_ATTENTION_SLIDING_WINDOW`, `LLM_KV_NEXTN_PREDICT_LAYERS`, `LLM_KV_ATTENTION_INDEXER_HEAD_COUNT`, `LLM_KV_ATTENTION_INDEXER_KEY_LENGTH`, and `LLM_KV_ATTENTION_INDEXER_TOP_K` already exist and can be reused for V4.

### 1.3 Add New Tensor Enums

**File:** `src/llama-arch.h`

Add to `llm_tensor`:

```cpp
// Low-rank O projection (DeepSeek V4)
LLM_TENSOR_ATTN_O_A,              // blk.%d.attn_o_a
LLM_TENSOR_ATTN_O_B,              // blk.%d.attn_o_b

// Hyper-Connections
LLM_TENSOR_HC_PRE,                // blk.%d.hc_pre
LLM_TENSOR_HC_POST,               // blk.%d.hc_post
LLM_TENSOR_HC_ATTN_FN,            // blk.%d.hc_attn_fn
LLM_TENSOR_HC_FFN_FN,             // blk.%d.hc_ffn_fn
LLM_TENSOR_HC_ATTN_BASE,          // blk.%d.hc_attn_base
LLM_TENSOR_HC_FFN_BASE,           // blk.%d.hc_ffn_base
LLM_TENSOR_HC_ATTN_SCALE,         // blk.%d.hc_attn_scale
LLM_TENSOR_HC_FFN_SCALE,          // blk.%d.hc_ffn_scale

// KV Compressor (per-layer, only where compress_ratio > 0)
LLM_TENSOR_COMPRESS_WKV,          // blk.%d.compress.wkv
LLM_TENSOR_COMPRESS_WGATE,        // blk.%d.compress.wgate
LLM_TENSOR_COMPRESS_APE,          // blk.%d.compress.ape
LLM_TENSOR_COMPRESS_NORM,         // blk.%d.compress.norm
```

**Note:** `LLM_TENSOR_ATTN_SINKS` and indexer tensors (`LLM_TENSOR_INDEXER_K_NORM`, `LLM_TENSOR_INDEXER_PROJ`, `LLM_TENSOR_INDEXER_ATTN_K`, `LLM_TENSOR_INDEXER_ATTN_Q_B`) already exist.

Add corresponding `LLM_TENSOR_NAMES` entries in `src/llama-arch.cpp` and `MODEL_TENSOR` entries in `gguf-py/gguf/constants.py`.

### 1.4 Register Tensors for DEEPSEEK4 Arch

**File:** `gguf-py/gguf/constants.py`

Add `MODEL_ARCH.DEEPSEEK4` tensor list (excerpt of key additions over `DEEPSEEK2`):

```python
MODEL_ARCH.DEEPSEEK4: [
    # embeddings / output
    MODEL_TENSOR.TOKEN_EMBD,
    MODEL_TENSOR.OUTPUT_NORM,
    MODEL_TENSOR.OUTPUT,
    # attention
    MODEL_TENSOR.ATTN_NORM,
    MODEL_TENSOR.ATTN_Q_A,
    MODEL_TENSOR.ATTN_Q_B,
    MODEL_TENSOR.ATTN_Q_A_NORM,
    MODEL_TENSOR.ATTN_KV_A_MQA,   # mapped from V4 "wkv"
    MODEL_TENSOR.ATTN_KV_A_NORM,
    MODEL_TENSOR.ATTN_O_A,        # NEW: low-rank O down-proj
    MODEL_TENSOR.ATTN_O_B,        # NEW: low-rank O up-proj
    MODEL_TENSOR.ATTN_OUT,
    MODEL_TENSOR.ATTN_SINKS,      # NEW: learned sink token
    # indexer (used when compress_ratio == 4)
    MODEL_TENSOR.INDEXER_K_NORM,
    MODEL_TENSOR.INDEXER_PROJ,
    MODEL_TENSOR.INDEXER_ATTN_K,
    MODEL_TENSOR.INDEXER_ATTN_Q_B,
    # compressor (used when compress_ratio > 0)
    MODEL_TENSOR.COMPRESS_WKV,
    MODEL_TENSOR.COMPRESS_WGATE,
    MODEL_TENSOR.COMPRESS_APE,
    MODEL_TENSOR.COMPRESS_NORM,
    # hyper-connections
    MODEL_TENSOR.HC_PRE,
    MODEL_TENSOR.HC_POST,
    MODEL_TENSOR.HC_ATTN_FN,
    MODEL_TENSOR.HC_FFN_FN,
    MODEL_TENSOR.HC_ATTN_BASE,
    MODEL_TENSOR.HC_FFN_BASE,
    MODEL_TENSOR.HC_ATTN_SCALE,
    MODEL_TENSOR.HC_FFN_SCALE,
    # FFN / MoE
    MODEL_TENSOR.FFN_NORM,
    MODEL_TENSOR.FFN_GATE_INP,
    MODEL_TENSOR.FFN_GATE_EXPS,
    MODEL_TENSOR.FFN_DOWN_EXPS,
    MODEL_TENSOR.FFN_UP_EXPS,
    MODEL_TENSOR.FFN_GATE_SHEXP,
    MODEL_TENSOR.FFN_DOWN_SHEXP,
    MODEL_TENSOR.FFN_UP_SHEXP,
    MODEL_TENSOR.FFN_EXP_PROBS_B,
],
```

Also add `ROPE_FREQS` to `MODEL_TENSOR_SKIP_LIST` if needed (same as `DEEPSEEK2`).

### 1.5 Python Converter Class

**File:** `convert_hf_to_gguf.py`

Add:
```python
@ModelBase.register("DeepseekV4ForCausalLM")
class DeepseekV4Model(DeepseekV2Model):
    model_arch = gguf.MODEL_ARCH.DEEPSEEK4
    skip_mtp = True  # same as V2/V3 until MTP is supported
```

Override `set_gguf_parameters` to write V4-specific keys:
- `hc_mult`, `hc_eps`
- `o_lora_rank`, `o_groups`
- `num_hash_layers`
- `compress_rope_theta`
- Per-layer `compress_ratios` array (write as `attention.layer_compress_ratio`)
- `index_n_heads`, `index_head_dim`, `index_topk`
- `scoring_func` → map to `expert_gating_func` (see Phase 2.4 for new enum)
- `swiglu_limit` → map to existing `swiglu_clamp_exp` (per-layer)

Override `modify_tensors`:
- Map HF `wkv` → GGUF `attn_kv_a_mqa` (same tensor enum, different HF source name)
- Map HF `wo_a` / `wo_b` → `attn_o_a` / `attn_o_b`
- Map HC tensors: `hc_attn_fn`, `hc_ffn_fn`, `hc_attn_base`, `hc_ffn_base`, `hc_attn_scale`, `hc_ffn_scale`, `hc_pre`, `hc_post`
- Map compressor tensors: `compressor.wkv`, `compressor.wgate`, `compressor.ape`, `compressor.norm`
- Map indexer tensors (same names as GLM_DSA): `indexer.k_norm`, `indexer.proj`, `indexer.attn_k`, `indexer.attn_q_b`
- Handle `attn_sink` tensor
- Skip MTP layers (same logic as V2/V3)
- Merge experts (same stacking logic as V2/V3)

---

## Phase 2: C++ Hyperparameters & Model Loading

### 2.1 Extend `llama_hparams`

**File:** `src/llama-hparams.h`

Add fields:

```cpp
// Hyper-Connections
uint32_t hc_mult = 1;
float    hc_eps  = 1e-6f;

// Low-rank O
uint32_t n_lora_o = 0;
uint32_t n_o_groups = 1;

// Hash routing for early MoE layers
uint32_t n_hash_layers = 0;

// KV compression
uint32_t compress_rope_theta = 160000;
std::array<uint32_t, LLAMA_MAX_LAYERS> compress_ratios; // 0 = no compression

// V4 MoE
float    swiglu_limit = 10.0f;
```

**Note:** `indexer_n_head`, `indexer_head_size`, `indexer_top_k`, `n_swa`, and `nextn_predict_layers` already exist.

### 2.2 Model Loader Parameter Parsing

**File:** `src/llama-model.cpp`

Add new `case LLM_ARCH_DEEPSEEK4:` alongside `LLM_ARCH_DEEPSEEK2` in the hparams loading switch.

Load keys:
```cpp
case LLM_ARCH_DEEPSEEK4:
{
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT, hparams.n_layer_dense_lead, false);
    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK, hparams.n_lora_q);
    ml.get_key(LLM_KV_ATTENTION_O_LORA_RANK, hparams.n_lora_o);
    ml.get_key(LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT, hparams.n_o_groups);
    ml.get_key(LLM_KV_HASH_LAYER_COUNT, hparams.n_hash_layers);
    ml.get_key(LLM_KV_HYPER_CONNECTION_MULT, hparams.hc_mult);
    ml.get_key(LLM_KV_HYPER_CONNECTION_EPS, hparams.hc_eps);
    ml.get_key(LLM_KV_ROPE_FREQ_BASE_COMPRESS, hparams.compress_rope_theta, false);
    ml.get_key_or_arr(LLM_KV_ATTENTION_COMPRESS_RATIO, hparams.compress_ratios, hparams.n_layer, false);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K, hparams.indexer_top_k);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa);
    ml.get_key(LLM_KV_SWIGLU_CLAMP_EXP, hparams.swiglu_limit, false); // per-layer if needed
    // ... existing MoE params (n_ff_exp, n_expert, n_expert_shared, etc.)
    // ... existing rope params (yarn, etc.)
}
```

### 2.3 Extend `llama_layer` Tensor Pointers

**File:** `src/llama-model.h`

Add to `llama_layer`:

```cpp
// Low-rank O (DeepSeek V4)
struct ggml_tensor * wo_a = nullptr;  // ColumnParallel down-proj
struct ggml_tensor * wo_b = nullptr;  // RowParallel up-proj (name collision: existing wo_b is attn_out bias!)
```

**IMPORTANT:** `llama_layer` already has `wo_b` (attention output bias). In V4, `wo_b` is a **weight** matrix for low-rank O. To avoid a naming collision, add `wo_a` and rename the V4 low-rank tensor to something like `wo_lora_a` / `wo_lora_b`:

```cpp
struct ggml_tensor * attn_o_a = nullptr;  // low-rank O down
struct ggml_tensor * attn_o_b = nullptr;  // low-rank O up
```

Add HC tensors:
```cpp
struct ggml_tensor * hc_pre        = nullptr;
struct ggml_tensor * hc_post       = nullptr;
struct ggml_tensor * hc_attn_fn    = nullptr;
struct ggml_tensor * hc_ffn_fn     = nullptr;
struct ggml_tensor * hc_attn_base  = nullptr;
struct ggml_tensor * hc_ffn_base   = nullptr;
struct ggml_tensor * hc_attn_scale = nullptr;
struct ggml_tensor * hc_ffn_scale  = nullptr;
```

Add Compressor tensors:
```cpp
struct ggml_tensor * compress_wkv  = nullptr;
struct ggml_tensor * compress_wgate = nullptr;
struct ggml_tensor * compress_ape  = nullptr;
struct ggml_tensor * compress_norm = nullptr;
```

### 2.4 Tensor Loading (`create_tensor` calls)

**File:** `src/llama-model.cpp`

Add new `case LLM_ARCH_DEEPSEEK4:` in the giant tensor-loading switch.

Key differences from `LLM_ARCH_DEEPSEEK2`:

1. **Attention Q:** Same low-rank path (`wq_a`, `wq_b`, `attn_q_a_norm`).
2. **Attention KV:** Load `wkv_a_mqa` tensor but from the mapped name `wkv` (handled in converter). Shape: `{n_embd, head_dim}` (512) instead of `{n_embd, kv_lora_rank + n_embd_head_qk_rope}`.
3. **No `wk_b` / `wv_b` / `wkv_b`:** V4 does NOT use V2/V3-style MLA absorption. K and V are directly `head_dim` sized.
4. **Low-rank O:** Load `attn_o_a` and `attn_o_b`.
   - `attn_o_a`: `{n_head * head_dim / n_o_groups, n_o_groups * n_lora_o}` = `{4096, 8192}`
   - `attn_o_b`: `{n_o_groups * n_lora_o, n_embd}` = `{8192, 4096}`
5. **Attn Sink:** Load `LLM_TENSOR_ATTN_SINKS` (shape `{n_head}` or `{1}`).
6. **Compressor (conditional):** Only load if `compress_ratios[il] > 0`.
   - `compress_wkv`: shape depends on ratio and window size
   - `compress_wgate`: gating weights
   - `compress_ape`: absolute positional embeddings for compressor
   - `compress_norm`: RMS norm
7. **Indexer (conditional):** Only load if `compress_ratios[il] == 4` (or when indexer is active).
   - Reuse same tensors as `GLM_DSA`: `indexer_k_norm`, `indexer_proj`, `indexer_attn_k`, `indexer_attn_q_b`
8. **HC tensors:** Load for every layer.
   - `hc_pre`: `{hc_mult * n_embd, n_embd}` (or transposed; verify from HF weights)
   - `hc_post`: `{n_embd, hc_mult * n_embd}`
   - `hc_attn_fn`: `{(2+hc_mult)*hc_mult, hc_mult*n_embd}`
   - etc.
9. **MoE:** Same expert tensor loading as DEEPSEEK2, but:
   - For `il < n_hash_layers`: MoE router logic will differ (hash-based, not learned).
   - `ffn_gate_inp` may not exist for hash-routed layers.

### 2.5 New Expert Gating Function

**File:** `src/llama-hparams.h`

Add to `llama_expert_gating_func_type`:
```cpp
LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS = 4,
```

**File:** `src/llama-graph.cpp` (`build_moe_ffn`)

Add case:
```cpp
case LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS:
    {
        probs = ggml_sqrt_softplus(ctx0, logits); // needs new GGML op or composite
    } break;
```

If `ggml_sqrt_softplus` does not exist, implement as composite:
```cpp
// sqrt(softplus(x)) = sqrt(log(1 + exp(x)))
probs = ggml_sqrt(ctx0, ggml_softplus(ctx0, logits));
```
(Check if `ggml_softplus` exists; if not, implement via `ggml_log`, `ggml_add`, `ggml_exp`.)

---

## Phase 3: Graph Builder Implementation

### 3.1 Decision: Extend `deepseek2.cpp` vs. New File

**Recommendation:** Create a **new file** `src/models/deepseek4.cpp` and class `llm_build_deepseek4`, because the differences from V2/V3 are substantial enough to avoid polluting the stable `deepseek2.cpp` with many `if (arch == LLM_ARCH_DEEPSEEK4)` branches.

However, factor out reusable helper methods in `llama-graph.cpp` where possible:
- `build_hc_residual(...)` for Hyper-Connections
- `build_kv_compressor(...)` for compression
- `build_indexer_select(...)` for top-k indexing

**File:** `src/models/models.h`
Add:
```cpp
struct llm_build_deepseek4 : public llm_graph_context {
    llm_build_deepseek4(const llama_model & model, const llm_graph_params & params);
};
```

### 3.2 Attention Graph (MLA-ish but simplified)

V4 attention graph differs from V2/V3:

```cpp
// 1. Low-rank Q (same as V2/V3)
q = ggml_mul_mat(ctx0, model.layers[il].wq_a, cur);
q = build_norm(q, model.layers[il].attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
q = ggml_mul_mat(ctx0, model.layers[il].wq_b, q);

// 2. Split Q into nope and rope
q_nope = ggml_view_3d(..., head_dim - qk_rope_head_dim, ...);
q_pe   = ggml_view_3d(..., qk_rope_head_dim, ...);
q_pe   = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, ...);

// 3. KV projection (DIFFERENT from V2/V3)
// wkv shape: [n_embd, head_dim] -> output [head_dim, n_tokens]
kv = ggml_mul_mat(ctx0, model.layers[il].wkv_a_mqa, cur); // rename internally or use wkv field
// Split KV into nope and rope if needed, or apply RoPE to entire KV
k_pe = ggml_view_2d(ctx0, kv, qk_rope_head_dim, n_tokens, ...);
k_pe = ggml_rope_ext(ctx0, k_pe, inp_pos, nullptr, ...);
k_nope = ggml_view_2d(ctx0, kv, head_dim - qk_rope_head_dim, n_tokens, ...);

// 4. Build Q/K/V heads for MQA
// Q: [head_dim, n_head, n_tokens]
// K: [head_dim, 1, n_tokens] (broadcast to all heads)
// V: [head_dim, 1, n_tokens]
Qcur = ggml_concat(ctx0, q_nope, q_pe, 0); // reshape to 3d
Kcur = ggml_concat(ctx0, k_nope, k_pe, 0);
Vcur = kv; // or reconstructed from k_nope/k_pe if stored separately

// 5. Attention (reuse build_attn with inp_attn_kv or custom)
cur = build_attn(inp_attn_kv,
    model.layers[il].wo, nullptr, nullptr,  // wo is NOT used directly when low-rank O is active
    Qcur, Kcur, Vcur,
    nullptr, model.layers[il].attn_sinks, nullptr,
    kq_scale, il);
```

**Low-rank O:** After `build_attn` returns `[n_head * head_dim, n_tokens]`, apply:
```cpp
// Reshape attn_out from [n_head*head_dim, n_tokens] to [n_head*head_dim/n_o_groups, n_o_groups, n_tokens]
// Or treat as 2d: [n_head*head_dim, n_tokens]
cur = ggml_mul_mat(ctx0, model.layers[il].attn_o_a, cur); // [n_o_groups*n_lora_o, n_tokens]
cur = ggml_mul_mat(ctx0, model.layers[il].attn_o_b, cur); // [n_embd, n_tokens]
```

Wait: `build_attn` internally applies `wo`. We need to either:
- Pass `wo = nullptr` to `build_attn` and apply `attn_o_a` / `attn_o_b` externally, OR
- Create a new `build_attn_mqa` that skips `wo` and returns raw attention output.

**Recommendation:** Pass `wo = nullptr` (and `wo_b = nullptr`, `wo_s = nullptr`) to `build_attn`, then apply low-rank O manually. Verify `build_attn` handles `wo == nullptr` gracefully (it should, or add a 3-line guard).

### 3.3 KV Cache Compression & Indexer

This is the most complex new component. The graph needs to support per-layer compression ratios.

**High-level flow for a layer with `compress_ratio > 0`:**

1. Before storing K/V to KV cache, pass them through the compressor.
2. The compressor downsamples the sequence dimension by `compress_ratio` using learned gated pooling.
3. If `compress_ratio == 4`, additionally run the indexer to select top-k compressed positions.
4. Store **compressed** K/V in the cache. During attention, the cache is already compressed, so no runtime overhead per token after the initial compression.

**Graph implementation sketch:**

```cpp
// Inside layer loop, before attention:
if (hparams.compress_ratios[il] > 0) {
    // On the first token of a new chunk, run compressor
    // For inference, this is tricky: we compress the KV *before* storing it
    // The KV cache stores compressed representations
    
    // Runtime shape: [head_dim, n_tokens]
    ggml_tensor * k_comp = ggml_mul_mat(ctx0, model.layers[il].compress_wkv, kv);
    // Apply gate
    ggml_tensor * gate = ggml_mul_mat(ctx0, model.layers[il].compress_wgate, kv);
    gate = ggml_sigmoid(ctx0, gate);
    k_comp = ggml_mul(ctx0, k_comp, gate);
    // Add positional emb
    k_comp = ggml_add(ctx0, k_comp, model.layers[il].compress_ape);
    // Pool (downsample seq dim by ratio)
    // GGML does not have a generic pooling op; may need custom op or reshape + mean
    k_comp = ggml_pool_1d(ctx0, k_comp, GGML_OP_POOL_AVG, ratio, ratio, 0);
    // Norm
    k_comp = build_norm(k_comp, model.layers[il].compress_norm, nullptr, LLM_NORM_RMS, il);
    
    // Store k_comp instead of raw k in KV cache
    // This requires KV cache to support variable-length per layer!
}
```

**KV Cache Impact:** llama.cpp's KV cache assumes uniform `n_embd_k_gqa` per layer. Compressed layers have a **shorter sequence length** in the cache, not smaller head dim. This is easier to support than variable head dims:
- The cache cell shape is `[n_embd_head_k * n_head_kv, n_seq]`.
- Compression reduces `n_seq` stored by `compress_ratio`.
- Need to store `compress_ratio` per layer in the KV cache metadata and adjust `kv_self.n` accordingly.

**Indexer (ratio == 4):**
The indexer selects top-k positions from the compressed cache. This is analogous to GLM_DSA's sparse attention but applied to KV indexing.

Since GLM_DSA already loads indexer tensors, we can study its graph usage. If GLM_DSA does not yet use them in `llm_build_deepseek2`, we need to implement:
1. Project query to indexer space using `indexer_attn_q_b`.
2. Project compressed KV keys using `indexer_attn_k`.
3. Score and select top-k (`indexer_top_k`) positions.
4. Mask attention to only those indexed positions (or gather them).

**Recommendation for Phase 3.3:** Implement compression first without indexer (treat `ratio == 4` as simple pooling). Add indexer in a follow-up PR to keep scope manageable.

### 3.4 Hyper-Connections Graph

HC replaces the residual `inpL = ggml_add(ctx0, cur, inpSA)` with a learned mixing.

**State management:** Instead of a single `inpL` (hidden state), maintain `hc_mult` copies.

```cpp
// Initialize: after embeddings, replicate to hc_mult copies
// Shape: [n_embd, hc_mult, n_tokens]
ggml_tensor * hc_state = ggml_repeat(ctx0, inpL, ggml_new_tensor_3d(ctx0, inpL->type, n_embd, hc_mult, n_tokens));

for (int il = 0; il < n_layer; ++il) {
    // Pre: reduce hc_mult copies -> single hidden state for this layer's input
    ggml_tensor * cur = ggml_mul_mat(ctx0, model.layers[il].hc_pre, hc_state);
    // shape: [n_embd, n_tokens]
    
    // ... run attention and FFN on cur ...
    
    // Post: expand 1 -> hc_mult via learned combination
    // hc_post shape: [hc_mult * n_embd, n_embd]
    ggml_tensor * new_hc = ggml_mul_mat(ctx0, model.layers[il].hc_post, cur);
    // reshape to [n_embd, hc_mult, n_tokens]
    new_hc = ggml_reshape_3d(ctx0, new_hc, n_embd, hc_mult, n_tokens);
    
    // Mixing: combine old hc_state and new_hc using Sinkhorn weights
    // hc_attn_fn / hc_ffn_fn define mixing weights
    // For simplicity, initial implementation can approximate Sinkhorn with softmax
    // Full Sinkhorn requires iterative normalization (not a native GGML op)
    
    hc_state = new_hc; // or mixed version
}

// Final: reduce to single hidden state for output
inpL = ggml_mul_mat(ctx0, model.layers[last].hc_pre, hc_state);
```

**Sinkhorn Algorithm:** The HF code uses `hc_sinkhorn_iters=20` iterations of row/col normalization on a `[hc_mult, hc_mult]` mixing matrix. In GGML, this requires a custom loop of `ggml_scale`, `ggml_sum_rows`, `ggml_div`, etc., or a new fused `ggml_sinkhorn` op.

**Recommendation:** For the initial implementation, approximate Sinkhorn with a single softmax over rows (or a small fixed number of iterations using `ggml_graph` nodes). Document that full Sinkhorn fidelity may require a new GGML op.

### 3.5 Hash-Based Routing (Early MoE Layers)

For layers `il < n_hash_layers`, the MoE router is not learned; it uses a deterministic hash (`tid2eid`) mapping token IDs to expert IDs.

**Implementation:**
- In the converter, precompute the `tid2eid` mapping and store it as a GGUF array or tensor.
- In the graph builder, for hash layers, skip `ffn_gate_inp` entirely.
- Use `ggml_get_rows` or a custom lookup to select experts based on input token IDs.
- Since the mapping is token-ID-based, it only works for the first token of a sequence or requires knowing the token ID at each position. For inference, this may be simplified: hash layers might actually hash the hidden state or position, not the raw token ID. Verify from HF `inference/model.py`.

**If it's truly token-ID-based:** This is incompatible with standard autoregressive caching where past tokens are not re-evaluated. The hash routing likely applies only to the *current* token being processed.

**Recommendation:** If hash routing is token-ID based, implement it only for the non-cached path (prompt processing). For token generation, fallback to standard routing or verify DeepSeek's exact behavior. Add a model hparam `n_hash_layers` and conditionally skip `build_moe_ffn` gate logic for those layers.

### 3.6 MoE with `sqrtsoftplus` and Clamp

In `build_moe_ffn`, pass the new gating func enum.

For `swiglu_limit`, clamp the SwiGLU activation:
```cpp
// Inside build_ffn or MoE expert, after gate*up:
cur = ggml_clamp(ctx0, cur, -swiglu_limit, swiglu_limit);
```

If `ggml_clamp` exists, use it. Otherwise, implement with `ggml_max` + `ggml_min`.

---

## Phase 4: KV Cache & Memory Adjustments

### 4.1 Per-Layer Sequence Length

**File:** `src/llama-kv-cache.h` / `src/llama-kv-cache.cpp`

The KV cache currently tracks a single `n` (sequence length) per cell. For V4, layers with `compress_ratio > 0` store fewer elements.

**Approach:** Store `compress_ratio` per layer in the KV cache metadata. When computing cache offsets or attention masks, divide the effective sequence length by `compress_ratio` for compressed layers.

**Simpler approach:** Store the compressed KV as if it were a normal KV with a shorter sequence. The cache manager doesn't need to know about compression if the graph builder already compressed the tensor before `ggml_kv_cache_update` (or equivalent). The only requirement is that `n_embd_k_gqa` and `n_embd_v_gqa` remain the same (head_dim is unchanged; only seq dim shrinks).

**Verification needed:** Does `llama_kv_cache` accept variable `n` per layer? Currently `kv_self.n` is a scalar. If we store compressed KV, `kv_self.n` must represent the **uncompressed** length for non-compressed layers and **compressed** length for compressed layers. This mismatch means we likely need per-layer `kv_self.n` or a separate tracking array.

**Recommendation:** Add `std::vector<uint32_t> n_compressed;` to the KV cache struct, populated from `hparams.compress_ratios`. Update `llama_kv_cache_find_cell`, `llama_kv_cache_seq_rm`, etc., to use the appropriate length. This is a medium-sized change.

### 4.2 Sliding Window Integration

V4 uses `sliding_window: 128`. The existing SWA infrastructure (`n_swa`, `swa_layers`, `LLAMA_SWA_TYPE_STANDARD`) can be reused. Set `hparams.n_swa = 128` and mark all layers as SWA layers (or follow the `compress_ratios` pattern).

---

## Phase 5: Quantization & Backend

### 5.1 FP8 / FP4 Weights

The config specifies native FP8 quantization (`"quant_method": "fp8"`). llama.cpp already supports FP8 (E4M3) via GGUF type `GGML_TYPE_F8E4M3` and block-quantized FP8. The converter should preserve FP8 scales (`weight_block_size: [128, 128]`).

For FP4 experts (`n_routed_experts` can be FP4), ensure the converter maps them to the appropriate GGML type (e.g., `GGML_TYPE_Q4_0` or a future `GGML_TYPE_FP4`). If native FP4 GGML type does not exist yet, quantize to `Q4_0`/`Q4_K_M` as a fallback and document the limitation.

### 5.2 Tensor Buffer Types

Add `LLM_TENSOR_INFO` entries for all new tensors in `src/llama-arch.cpp`:
```cpp
{LLM_TENSOR_ATTN_O_A,       {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
{LLM_TENSOR_ATTN_O_B,       {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
{LLM_TENSOR_HC_PRE,         {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
{LLM_TENSOR_HC_POST,        {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
{LLM_TENSOR_HC_ATTN_FN,     {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
// ... etc
```

---

## Component Reuse vs. New Implementation Matrix

| Component | Existing Code to Reuse | New Code Required |
|-----------|----------------------|-------------------|
| **Low-rank Q** | `DEEPSEEK2` (`wq_a`, `wq_b`, `attn_q_a_norm`) | None |
| **YaRN RoPE** | `DEEPSEEK2` rope scaling logic | None |
| **MoE Backbone** | `DEEPSEEK2` (`build_moe_ffn`, expert merging, shared experts) | New gating func `sqrtsoftplus`; hash routing for first N layers |
| **MTP** | Skip logic in converter (`skip_mtp`) | None |
| **Indexer Tensors** | `GLM_DSA` (`indexer_k_norm`, `indexer_proj`, etc.) | Graph integration for top-k selection |
| **Attn Sink** | `OPENAI_MOE` (`attn_sinks` passed to `build_attn`) | None |
| **Sliding Window** | Existing `n_swa` / `swa_layers` | None |
| **FP8 Quant** | Existing FP8 E4M3 support in GGML | Scale format `ue8m0` mapping if needed |
| **Low-rank O** | None | New tensors `attn_o_a/b`, graph integration |
| **KV Compressor** | None | New tensors, new graph subgraph, KV cache seq length handling |
| **Hyper-Connections** | None | Entirely new residual path, Sinkhorn mixing, multi-state buffer |
| **Hash Routing** | None | Deterministic expert selection, `tid2eid` mapping |
| **Per-layer Compress Ratios** | None | Per-layer config array, conditional tensor loading |

---

## Phase 6: Testing & Validation Plan

1. **Converter Test:**
   - Run `convert_hf_to_gguf.py` on `deepseek-ai/DeepSeek-V4-Flash`.
   - Verify all V4-specific tensors appear in GGUF with `gguf-dump.py`.
   - Verify no MTP tensors are present (or are skipped).

2. **Model Loading Test:**
   - Load the GGUF with `llama-model-test` or a minimal binary.
   - Verify tensor count matches expected.
   - Verify `llama_model_n_layers()`, `llama_model_n_head()`, etc.

3. **Graph Construction Test:**
   - Build graph for `n_tokens = 1` and `n_tokens = 128`.
   - Use `LLAMA_GRAPH_INPUT_DEBUG=1` to inspect tensor shapes.

4. **Numerical Sanity:**
   - Compare logits of first few tokens against HF reference (allowing for FP8→FP16 conversion differences).
   - Focus on hash layers (first 3) and compressed layers (check `compress_ratios` array).

5. **Performance Baseline:**
   - Compare prefill and decode speed against V3 (should be faster due to smaller KV cache from compression).

---

## Summary of Files to Modify

| File | Nature of Change |
|------|-----------------|
| `gguf-py/gguf/constants.py` | Add `DEEPSEEK4` arch, new `MODEL_TENSOR`s, tensor lists |
| `convert_hf_to_gguf.py` | Add `DeepseekV4Model` class, tensor mapping, param writing |
| `src/llama-arch.h` | Add `LLM_ARCH_DEEPSEEK4`, new `llm_kv`, new `llm_tensor` enums |
| `src/llama-arch.cpp` | Add name mappings and tensor info entries |
| `src/llama-hparams.h` | Add V4 hparams (`hc_mult`, `n_lora_o`, `compress_ratios`, etc.) |
| `src/llama-model.h` | Add new tensor pointers to `llama_layer` |
| `src/llama-model.cpp` | Add `DEEPSEEK4` loader case, `create_tensor` calls, hparam parsing |
| `src/llama-model.cpp` (builder dispatch) | Add `LLM_ARCH_DEEPSEEK4` → `llm_build_deepseek4` |
| `src/llama-graph.cpp` / `.h` | Add `sqrtsoftplus` gating, new helper methods (HC, compressor) |
| `src/models/models.h` | Declare `llm_build_deepseek4` |
| `src/models/deepseek4.cpp` | **New file:** full graph builder for V4 |
| `src/llama-kv-cache.h/cpp` | Per-layer compressed sequence length tracking |
| `docs/development/HOWTO-add-model.md` | Update if new patterns (HC, compressor) are added |

---

## Open Questions / Risks

1. **Sinkhorn in GGML:** 20 iterations of row/col softmax normalization is expensive in a static graph. A custom `ggml_sinkhorn` op may be necessary for competitive performance.
2. **KV Cache Variable Length:** llama.cpp's KV cache is heavily optimized around a uniform `n`. Making it per-layer is a non-trivial refactor.
3. **Indexer Top-k on GPU:** The indexer selects top-k from compressed KV. `ggml_argsort_top_k` exists but may need tuning for this use case.
4. **Hash Routing at Decode Time:** If hash routing depends on raw token IDs, it conflicts with the standard assumption that the model only sees hidden states during generation. Clarify whether the hash is over hidden states or token IDs.
5. **Low-rank O + `build_attn`:** `build_attn` currently applies `wo` internally. Passing `wo=nullptr` and applying `attn_o_a/b` externally must be verified to not break attention masking or KV cache updates.
