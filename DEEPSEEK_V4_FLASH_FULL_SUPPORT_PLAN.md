# DeepSeek V4 Flash Full Support Plan

This plan covers the remaining work to support the official `deepseek-ai/DeepSeek-V4-Flash` checkpoint end to end in `llama.cpp`.

## Reference Checkpoint

Key configuration values from the official Hugging Face checkpoint:

- `num_hidden_layers = 43`
- `num_nextn_predict_layers = 1`
- `num_hash_layers = 3`
- `head_dim = 512`
- `num_attention_heads = 64`
- `num_key_value_heads = 1`
- `q_lora_rank = 1024`
- `o_lora_rank = 1024`
- `o_groups = 8`
- `n_routed_experts = 256`
- `num_experts_per_tok = 6`
- `scoring_func = "sqrtsoftplus"`
- `sliding_window = 128`
- `index_n_heads = 64`
- `index_head_dim = 128`
- `index_topk = 512`
- `hc_mult = 4`
- `hc_eps = 1e-6`
- `hc_sinkhorn_iters = 20`
- `compress_rope_theta = 160000`
- `compress_ratios = [0, 0, 4, 128, 4, 128, ..., 4, 0]`

Primary references:

- `config.json`
- `inference/model.py`
- `inference/convert.py`

## Current State

The current branch supports only the plain-attention / plain-MoE subset.

The current code explicitly rejects:

- MTP / NextN
- hash-routed MoE layers
- hyper-connections
- compressed sparse attention
- compressed-attention indexer

Those rejects live in:

- [convert_hf_to_gguf.py](./convert_hf_to_gguf.py)
- [src/llama-model.cpp](./src/llama-model.cpp)

Grouped `wo_a` / `wo_b` projection has already been corrected and should not be treated as pending work.

## Phase 0: Make Official Checkpoint Convertible

Goal: stop hard-failing on the official checkpoint and produce a structurally complete GGUF.

Tasks:

- Extend GGUF schema and tensor mappings for missing DeepSeek V4 Flash tensors:
  - `attn_sink`
  - compressor tensors
  - indexer tensors
  - hash-routing table `tid2eid`
  - hyper-connection tensors: `hc_attn_*`, `hc_ffn_*`, `hc_head_*`
- Update DeepSeek4 conversion logic so `block_count = num_hidden_layers + num_nextn_predict_layers`
- Add DeepSeek4 NextN tensors to `MODEL_TENSORS`
- Remove converter-side hard rejects once the tensors can be represented
- Confirm whether the stock HF FP8 checkpoint can be converted directly
- If the stock checkpoint needs preprocessing, add the minimum converter path needed to absorb the layout used by DeepSeek's `inference/convert.py`

Deliverable:

- The official HF checkpoint converts into a GGUF without requiring manual surgery

## Phase 1: Load-Time Support for All Flash Tensors

Goal: teach the runtime to load every tensor emitted by the converter.

Tasks:

- Add new `llama_layer` fields for:
  - hash-routing tensor(s)
  - compressor weights
  - hyper-connection weights
  - any additional per-layer sparse-attention state needed by Flash
- Extend `llama-arch.*` names and tensor metadata as needed
- Extend DeepSeek4 `load_hparams()` and `load_tensors()` in `src/llama-model.cpp`
- Keep the fixed grouped `wo_a` / `wo_b` layout intact

Deliverable:

- GGUF loads successfully with all DeepSeek V4 Flash tensors present

## Phase 2: Implement Compressed Sparse Attention

Goal: support the main Flash attention path.

This is the largest missing runtime feature.

Required behavior from the reference implementation:

- pure sliding-window attention when `compress_ratio == 0`
- sliding-window attention plus compressed KV when `compress_ratio > 0`
- `attn_sink`
- compressed-RoPE base using `compress_rope_theta`
- ratio `4` overlap compressor
- ratio `128` non-overlap compressor
- indexer path on the layers that use it

Tasks:

- Add DeepSeek4 graph logic for compressed KV cache construction
- Integrate `attn_sink`
- Reuse or adapt existing `attn_sinks` and `indexer_*` scaffolding already present in the repo
- Implement top-k selection over compressed memory
- Validate long-context behavior across:
  - pure SWA layers
  - ratio `4` layers
  - ratio `128` layers

Deliverable:

- DeepSeek4 attention matches the reference architecture for compressed and uncompressed layers

## Phase 3: Implement Hash-Routed MoE for First `num_hash_layers`

Goal: support the first 3 Flash layers that use token-id-based routing.

The official model uses `tid2eid[input_ids]` for the first `num_hash_layers`, not normal score-based top-k routing.

Tasks:

- Add tensor type and mapping for `mlp.gate.tid2eid`
- Store the tensor in the runtime
- Extend the MoE path so:
  - layers `< n_hash_layers` use token-id lookup
  - later layers use the existing routed-gating path
- Preserve existing `sqrtsoftplus` gating behavior for non-hash layers

Deliverable:

- Early Flash MoE layers behave like the official model instead of dense FFN or standard routed MoE

## Phase 4: Implement Hyper-Connections

Goal: replace plain residual adds with the Flash hyper-connection path.

The official model uses hyper-connections in every block plus an HC output head.

Tasks:

- Add tensor mappings for `hc_attn_*`, `hc_ffn_*`, and `hc_head_*`
- Load those tensors in the DeepSeek4 runtime
- Replace the current plain residual flow in `src/models/deepseek4.cpp` with:
  - HC split/mix before attention
  - HC combine after attention
  - HC split/mix before FFN
  - HC combine after FFN
  - HC head at the model output
- Decide whether to implement the exact sinkhorn path first, or a faithful fallback path that preserves model semantics

Deliverable:

- DeepSeek4 residual flow matches the reference HC architecture

## Phase 5: Implement MTP / NextN

Goal: support the extra NextN prediction layer used by Flash.

The repo already has generic NextN tensor ids and `llama_layer_nextn` storage. DeepSeek4 just does not use them yet.

Tasks:

- Add DeepSeek4 converter support for the extra NextN tensors
- Load NextN tensors for DeepSeek4
- Add a DeepSeek4 MTP / NextN graph path after the base transformer stack
- Ensure layer counting and tensor indexing work with:
  - `num_hidden_layers = 43`
  - `num_nextn_predict_layers = 1`

Deliverable:

- Full Flash checkpoint layout is represented and executable, including the MTP layer

## Phase 6: Validation and Regression Tests

Goal: verify behavior against the official implementation and prevent regressions.

Validation:

- Convert the official HF checkpoint end to end
- Load it in `llama.cpp`
- Compare logits against the official reference implementation for:
  - a short prompt
  - a prompt crossing the sliding-window boundary
  - a long-context prompt hitting compressed KV
  - a prompt exercising hash-routed MoE
  - a prompt with the MTP layer active

Regression coverage:

- grouped `wo_a` layout
- hash-routing `tid2eid`
- compressor ratio `4`
- compressor ratio `128`
- HC-enabled block smoke test
- MTP tensor loading and graph construction

Deliverable:

- Logit parity confidence and automated regression protection

## Recommended Execution Order

1. Phase 0: converter/schema/tensor mapping
2. Phase 1: loader support
3. Phase 2: compressed sparse attention
4. Phase 3: hash-routed MoE
5. Phase 4: hyper-connections
6. Phase 5: MTP / NextN
7. Phase 6: validation and regression tests

## Notes

- The main technical risk is compressed sparse attention, not grouped output projection.
- Hyper-connections are architectural and should not be deferred if the goal is real Flash support rather than approximate support.
- MTP is additive and can be implemented after base Flash inference is working, but full checkpoint support still requires it.
