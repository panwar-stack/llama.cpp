# DeepSeek-V4-Flash Support Implementation Summary

## Multi-Agent Workflow

- **Agent 1** analyzed the DeepSeek-V4 architecture (config.json, inference guide, whitepaper references) and produced a detailed implementation plan covering: new arch enum, GGUF schema extensions, converter changes, C++ hparams/tensor loading, and a custom graph builder.
- **Agent 2** verified the plan against the actual llama.cpp codebase, correcting several assumptions (e.g., V4 does not use MLA absorption, needs direct MQA, low-rank grouped O, `sqrtsoftplus` gating, and fundamentally different KV handling). It identified that hyper-connections, KV compression, and hash routing are extremely complex and should be deferred for a minimal viable implementation.
- **Agent 3** implemented the verified plan with focused, minimal changes.
- **Agent 4** built the project, ran Python and C++ tests, and confirmed no regressions.

---

## What Was Implemented

### 1. Python Converter & GGUF Schema
- **`gguf-py/gguf/constants.py`**
  - Added `MODEL_ARCH.DEEPSEEK4`
  - Added new tensor types: `ATTN_O_A`, `ATTN_O_B`
  - Added new metadata keys: `O_LORA_RANK`, `OUTPUT_GROUP_COUNT`, `HASH_LAYER_COUNT`, `HYPER_CONNECTION_MULT`, `HYPER_CONNECTION_EPS`, `ROPE_FREQ_BASE_COMPRESS`, `COMPRESS_RATIO`
- **`gguf-py/gguf/tensor_mapping.py`**
  - Mapped HF `self_attn.wkv` -> `ATTN_KV_A_MQA`
  - Mapped HF `self_attn.wo_a_proj` -> `ATTN_O_A`
  - Mapped HF `self_attn.wo_b_proj` -> `ATTN_O_B`
- **`convert_hf_to_gguf.py`**
  - Added `DeepseekV4Model` registered for `DeepseekV4ForCausalLM`
  - Writes V4-specific params: `head_dim` as key/value length, `o_lora_rank`, `o_groups`, `sqrtsoftplus` gating, `compress_rope_theta`, `compress_ratios`, `num_hash_layers`, indexer dims, hyper-connection dims, `swiglu_limit`

### 2. C++ Architecture & Loading
- **`src/llama-arch.h` / `src/llama-arch.cpp`**
  - Added `LLM_ARCH_DEEPSEEK4`, tensor names, KV key strings, and tensor op infos
- **`src/llama-hparams.h`**
  - Added `n_lora_o`, `n_o_groups`, `n_hash_layers`, `hc_mult`, `hc_eps`, `rope_freq_base_compress`, `compress_ratios`
  - Added `LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS = 4`
- **`src/llama-model.h`**
  - Added `attn_o_a` and `attn_o_b` to `llama_layer`
- **`src/llama-model.cpp`**
  - Hparam loader for DEEPSEEK4 (reads all V4-specific metadata)
  - Tensor loader with correct shapes for direct MQA KV (`head_dim` instead of `kv_lora_rank`), low-rank grouped O, and conditional MoE/dense FFN for hash layers
  - Graph dispatch to `llm_build_deepseek4`
  - RoPE type registration
  - Logging for V4-specific hyperparameters

### 3. Graph Builder (`src/models/deepseek4.cpp`)
Implements the core forward pass with these V4-specific features:
- **Low-rank Q** (same pattern as DeepSeek-V2/V3)
- **Direct MQA KV** (`wkv` projects directly to `head_dim = 512`; no MLA absorption)
- **Low-rank grouped O projection** (`wo_a` applied per-group, then `wo_b`, then summed)
- **Standard MoE** with `sqrtsoftplus` gating + shared expert
- **Hash-layer fallback**: layers without `ffn_gate_inp` fall back to dense FFN

### 4. MoE Gating
- **`src/llama-graph.cpp`**
  - Added `LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS` case: `probs = sqrt(softplus(logits))`

---

## Known Limitations / TODOs

The following advanced V4 features are **not yet implemented** and are left as TODOs for future incremental work:
1. **Hyper-Connections (HC)** -- replaces standard residuals with learned multi-state mixing + Sinkhorn normalization
2. **KV Compression & Indexer** -- per-layer `compress_ratios` with learned compressors and top-k indexers for sparse attention
3. **Hash-based Routing** -- deterministic `tid2eid` lookup for the first `num_hash_layers` layers
4. **MTP (Multi-Token Prediction)** -- skipped (same as V2/V3)
5. **Custom sparse attention kernel** -- long-context sparse attention is not yet optimized

---

## Tests & Validation
- **Build**: `llama` library compiles cleanly with no warnings
- **Python**: `gguf-py` unit tests pass
- **C++**: `test-gguf`, `test-chat-template`, `test-grammar-parser` all pass
- **No regressions** detected in existing architectures
