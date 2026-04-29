# DeepSeek V4 Flash — Learning & Onboarding Guide

Companion to commit `edc1d49ee` ("deepseek : add initial support for DeepSeek-V4-Flash architecture"). Read alongside [DEEPSEEK_V4_FLASH_FULL_SUPPORT_PLAN.md](./DEEPSEEK_V4_FLASH_FULL_SUPPORT_PLAN.md), which describes the staged plan; this guide describes what actually landed in this commit and how to study it.

---

## 1. What this commit does, in one paragraph

It introduces a new architecture id `LLM_ARCH_DEEPSEEK4` (string `"deepseek4"`) end-to-end: GGUF schema (Python writer + C++ reader), an HF-to-GGUF converter `DeepseekV4Model`, runtime hyperparameters, per-layer tensor allocation, a new graph builder `llm_build_deepseek4`, a new ggml op `GGML_OP_SPARSE_ATTN` with a CPU kernel, plus three test surfaces (a Python logits-parity harness, a C++ tensor-catalog test, and a C++ KV-validation test). What is **not** here yet: a real top-k indexer for ratio-4 layers, persistent compressed-KV state for decode, and the SIMD/GPU implementations of `GGML_OP_SPARSE_ATTN`. Those are listed in the plan doc as Phases 2/4.

The commit covers Phase 0–Phase 5 from the plan in skeleton form. Most paths exist and execute; some take simplified fallbacks marked with `TODO:`.

---

## 2. Concepts you must understand before reading the code

DeepSeek V4 Flash differs from V3/V2 along five axes. If you do not know what each of these is, the diff will look like noise.

| # | Concept | Where it appears in the diff |
|---|---|---|
| 1 | **Hyper-Connections (HC)** — replaces `x = x + f(x)` residuals with a learned `(2+m)·m`-channel mixing. | `build_hc_pre`, `build_hc_post`, `build_hc_head` in `src/models/deepseek4.cpp` |
| 2 | **Compressed Sparse Attention** — keeps a sliding window of full-resolution KV and a coarser pooled KV stream (ratio 4 overlap, ratio 128 non-overlap), gated by a learned compressor. | `attn_compressor_*` tensors; the `compress_ratio > 0` branch in `deepseek4.cpp` |
| 3 | **Indexer + top-k** — a small auxiliary attention scores tokens so the main attention only attends to top-k. The new ggml op `GGML_OP_SPARSE_ATTN` exists for this. | `ggml_sparse_attn` in `ggml/include/ggml.h`, `ggml/src/ggml.c`, kernel in `ggml-cpu/ops.cpp` |
| 4 | **Hash-routed MoE** — the first `n_hash_layers` MoE blocks use a fixed `tid2eid[input_ids]` table instead of score-based top-k. | `FFN_GATE_TID2EID`, the `selected_experts_in` parameter on `build_moe_ffn` |
| 5 | **Grouped low-rank output projection (`wo_a` / `wo_b`)** — the per-head output projection is split into a group-wise rank-`o_lora_rank` factorization. | `attn_o_a`, `attn_o_b` tensors; the o-projection branch in `deepseek4.cpp` |
| 6 | **`sqrtsoftplus` gating** — new MoE scoring function: `sqrt(softplus(logits))`. | `LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS` in `llama-hparams.h`, applied in `llama-graph.cpp` |
| 7 | **MTP / NextN** — the model emits an extra "next-next-token" prediction head from the last hidden state plus an embedding of the current token. | `nextn_predict_layers` loop at the bottom of `deepseek4.cpp` |
| 8 | **Direct MQA (no MLA)** — V4 has `n_kv = 1` and projects KV directly to `head_dim`. There is no `kv_b_proj`. | `wkv_a_mqa` shape `{n_embd, n_embd_head_k}` in `llama-model.cpp` |

References (paper + reference impl) are linked from the Plan doc.

---

## 3. The file map

The commit touches three layers. Read them in this order:

```
gguf-py/gguf/constants.py        ← the schema (start here)
gguf-py/gguf/gguf_writer.py      ← writer helpers for new keys
gguf-py/gguf/tensor_mapping.py   ← HF-name → GGUF-name aliases
convert_hf_to_gguf.py            ← DeepseekV4Model + FP8 dequant
                  ↓ produces a .gguf

src/llama-arch.h / .cpp          ← C++ mirror of the schema
src/llama-hparams.h              ← hparam fields
src/llama-model.h                ← per-layer tensor handles
src/llama-model.cpp              ← load_hparams + load_tensors
src/llama-context.cpp            ← node-budget bump
                  ↓ loads the .gguf

src/models/models.h              ← forward-decl
src/models/deepseek4.cpp         ← graph builder (the heart of the change)
src/llama-graph.cpp / .h         ← build_moe_ffn(selected_experts), build_sparse_attn
                  ↓ runs the graph

ggml/include/ggml.h              ← ggml_sparse_attn op
ggml/src/ggml.c                  ← op metadata + factory
ggml/src/ggml-backend-meta.cpp   ← backend split-state
ggml/src/ggml-cpu/ops.h          ← CPU kernel decl
ggml/src/ggml-cpu/ops.cpp        ← CPU kernel impl
ggml/src/ggml-cpu/ggml-cpu.c     ← dispatch + thread plan

tests/CMakeLists.txt             ← register tests
tests/test-deepseek4.cpp         ← tensor catalog round-trip
tests/test-deepseek4-validate.cpp← KV-key validation
tests/test-deepseek4-logits.py   ← parity vs. HF transformers
```

---

## 4. Guided tour — phase by phase

### 4.1 Phase 0: schema + converter

Goal: parse the official HF FP8 checkpoint and emit a structurally-complete GGUF.

**`gguf-py/gguf/constants.py`** — adds:
- One arch enum `DEEPSEEK4` and its name `"deepseek4"`.
- New KV keys: `HASH_LAYER_COUNT`, `HC_MULT/EPS/SINKHORN_ITERS`, `O_LORA_RANK`, `OUTPUT_GROUP_COUNT`, `COMPRESS_RATIO`, `FREQ_BASE_COMPRESS`.
- New `MODEL_TENSOR` ids: `ATTN_O_A/B`, `ATTN_COMPRESSOR_{APE,NORM,WGATE,WKV}`, `FFN_GATE_TID2EID`, `INDEXER_COMPRESSOR_*`, `HC_{ATTN,FFN,HEAD}_{FN,BASE,SCALE}`, `NEXTN_HC_HEAD_*`.
- A `MODEL_TENSORS[DEEPSEEK4]` list — read this first; it is the canonical inventory of every tensor a Flash GGUF must contain.
- `ExpertGatingFuncType.SQRT_SOFTPLUS = 4`.

**`gguf-py/gguf/gguf_writer.py`** — wires `add_hash_layer_count`, `add_hc_*`, `add_attention_compress_ratios`, `add_rope_freq_base_compress`. Mechanical.

**`gguf-py/gguf/tensor_mapping.py`** — adds *two* HF-name aliases per new tensor:
1. The stock HF naming, e.g. `model.layers.{bid}.self_attn.wo_a`.
2. The naming used by DeepSeek's own `inference/convert.py`, e.g. `layers.{bid}.attn.wo_a`.

This is why `tensor_mapping.py` doubles in size — every Flash tensor needs both spellings.

**`convert_hf_to_gguf.py`** — the new `DeepseekV4Model` class (registered for `DeepseekV4ForCausalLM`):

| Concern | What the code does |
|---|---|
| Block count | `block_count = num_hidden_layers + num_nextn_predict_layers` (so MTP gets a layer slot). |
| MLA-skip | Subclasses `DeepseekV2Model` but calls `TextModel.set_gguf_parameters(self)` directly to bypass V2's MLA logic. |
| `kv_b_proj` | Skipped — V4 has no MLA factorization. |
| `wo_a` reshape | Stored flat in HF; reshaped to `(o_groups, o_lora_rank, in_dim)` so the runtime can treat it as a batched matmul. |
| MTP `e_proj`/`h_proj` | DeepSeek splits these in HF; the converter concatenates them into the existing `eh_proj` schema. |
| Inference-format experts | DeepSeek's `inference/convert.py` lays out experts as `layers.{N}.ffn.experts.{E}.w{1,2,3}.weight`; the converter merges them per-layer into stacked `w1/w2/w3` tensors. |
| `attn_sink` | Added as a bare parameter in HF; the converter appends `.weight` so it matches the runtime's `ATTN_SINKS` spelling. |

Also in this file: a real **FP8 dequantizer** (`fp8_e4m3fn_to_float`, `fp8_e8m0_to_float`) is added on the `dequant_simple` path. Necessary because the official Flash checkpoint is FP8. Note the `LazyTorchTensor` changes: the byteswap map and safetensors map both grew an FP8 lane, gated with `getattr(torch, "float8_e4m3fn", torch.uint8)` for older torch versions.

### 4.2 Phase 1: C++ schema mirror + tensor allocation

**`src/llama-arch.h` / `.cpp`** — the literal C++ counterpart of `constants.py`. Every `LLM_KV_*` and `LLM_TENSOR_*` is mirrored with:
- a name in `LLM_KV_NAMES` / `LLM_TENSOR_NAMES`,
- a layer-class entry in `LLM_TENSOR_INFOS` (e.g. `LAYER_REPEATING` vs. `LAYER_OUTPUT`, and the `GGML_OP_*` it expects).

A subtle change: `NEXTN_*` tensors moved from `LAYER_OUTPUT` to `LAYER_REPEATING`. NextN now lives in real (per-block) layer slots, not the output bucket — required for the per-block MTP graph in `deepseek4.cpp`.

**`src/llama-hparams.h`** — adds `n_lora_o`, `n_o_groups`, `n_hash_layers`, `hc_mult/eps/sinkhorn_iters`, `rope_freq_base_compress`, and a fixed-size `compress_ratios[LLAMA_MAX_LAYERS]` (per-layer).

**`src/llama-model.h`** — extends `llama_layer` with handles for: `attn_o_a/b`, `attn_compressor_*`, `indexer_compressor_*`, `hc_attn_*`, `hc_ffn_*`, `ffn_gate_tid2eid`. Extends `llama_layer_nextn` with `hc_head_*`. Adds three top-level `hc_head_*` on `llama_model` for the global HC head.

**`src/llama-model.cpp`** — `LLM_ARCH_DEEPSEEK4` cases in:
- `load_hparams`: pulls every key, defaults `expert_gating_func` to `SQRT_SOFTPLUS` if not present, special-cases `n_layer ∈ {43,44}` for type tagging.
- `load_tensors`: the long loop is the runtime equivalent of `MODEL_TENSORS[DEEPSEEK4]`; learn it by walking it once next to the constants list. Notable shapes:
  - `wkv_a_mqa: {n_embd, n_embd_head_k}` — direct projection, not factorized.
  - `attn_o_a: {n_head*n_embd_head_v / o_groups, o_lora_rank, o_groups}` — 3D, batched on `o_groups`.
  - `attn_compressor_ape: {coff*head_dim, compress_ratio}` where `coff = 2` for ratio 4, else `1`.
  - `hc_attn_fn: {hc_dim, mix_hc}` with `mix_hc = (2 + hc_mult) * hc_mult`.
  - Indexer tensors only created when `compress_ratio == 4`.
  - NextN tensors only created for the last `nextn_predict_layers` slots.

**`src/llama-context.cpp`** — bumps `graph_max_nodes` for `LLM_ARCH_DEEPSEEK4` to `max(2048, 16·n_tensors())`. Necessary because the HC + sparse paths produce a much larger graph than DeepSeek V2.

### 4.3 Phase 2/4: graph builder (`src/models/deepseek4.cpp`)

This is the most important file in the commit. Three local helpers, then the constructor.

**`build_hc_pre(x, hc_fn, hc_base, hc_scale, ...) → (cur, post, comb)`**
- Computes the `(2+m)·m`-channel mixing tensor from `x` after RMS-norming over the `(hc_dim = m·n_embd)` axis.
- Splits the channels into three groups:
  - `pre[m]` → sigmoid + eps, used to weight which of the `m` HC-streams to read.
  - `post[m]` → 2·sigmoid, used by `build_hc_post` to weight the new value.
  - `comb[m, m]` → softmax + Sinkhorn iterations, the doubly-stochastic mixing matrix used to recombine streams.
- Returns the reduced 2D `result = sum_i pre[i] * x[:,i,:]`, plus `post` and `comb` for the matching `build_hc_post`.

**`build_hc_post(x, residual, post, comb)`**
- Combines `x` and the residual via `term1 = post * x` plus `term2 = comb @ residual` (per-token `m × m` matmul). This is the Flash replacement for `x + residual`.

**`build_hc_head(x, hc_head_fn, hc_head_base, hc_head_scale, ...)`**
- Applied once at the model output: simpler than `hc_pre` (only the `pre` channel), it collapses the `m`-stream representation back to a single stream before the final RMS-norm and `lm_head`.

**`llm_build_deepseek4` constructor** — the main loop:
1. Embed tokens, then if `hc_mult > 0` expand to `{n_embd, hc_mult, n_tokens}` by repeat.
2. For each of `n_layer - nextn_predict_layers` layers:
   - **HC pre vs. plain RMSNorm** (gated on `hc_mult`).
   - **Q projection**: lora-factored if `q_lora_rank > 0`, else plain.
   - **KV projection**: V4 MQA, single head, `wkv_a_mqa` then split nope/rope.
   - **Attention path**:
     - If `compress_ratio > 0` and `n_tokens >= compress_ratio`: Flash compressed sparse path — write local KV to cache; build a compressed KV stream via `attn_compressor_wgate/wkv` + APE + softmax-pooling; concat local + compressed; run `build_attn_mha` on the concat. Note the `coff == 2` branch currently takes a `// TODO: implement proper overlap transform` shortcut (drops half the channels).
     - Else: standard `build_attn`.
   - **Output projection**: grouped (`attn_o_a` 3D matmul → permute → flatten → `attn_o_b`) when both factors exist, else plain `wo`.
   - **HC post or plain residual**.
   - **FFN path**:
     - Dense if `il < n_layer_dense_lead`.
     - Else MoE. **First `n_hash_layers` layers** call `build_moe_ffn(..., selected_experts = get_rows(ffn_gate_tid2eid, t_inp_tokens))` to bypass gating; subsequent layers use the standard score-based path.
     - Shared expert is added in.
   - **HC post or plain residual** for the FFN.
3. After the loop, optional `build_hc_head` then RMS-norm and `lm_head` to logits.
4. **MTP / NextN block** for each `nextn_predict_layers`:
   - Embed current tokens, RMS-norm them (`enorm`); RMS-norm the hidden state (`hnorm`).
   - Split the joint `eh_proj` into `e_proj` and `h_proj` views; combine `e_proj(e) + h_proj(x)` (with broadcasting over the HC stream dim).
   - Run an attention + MoE block (re-using the same layer tensors) with HC pre/post.
   - Apply per-layer `hc_head` then `shared_head_norm` then `shared_head_head` to produce the MTP logits, expanded into the forward graph.

A useful mental model: the main loop is the V4 transformer; the trailing block is the V4 NextN head, sharing layer storage but with its own NextN-specific tensors.

### 4.3 (cont.) `src/llama-graph.cpp` / `.h`

Two pieces:

1. **`build_moe_ffn` overload extended** with `selected_experts_in`. When supplied, the function bypasses logits/probs/topk and treats `probs` as uniform 1.0 (so the existing post-selection mixing math just acts as identity-weighted dispatch).
2. **`LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS`** added to the gating switch: `probs = sqrt(softplus(logits))`.
3. **`build_sparse_attn`** wrapper around `ggml_sparse_attn`, doing the BHND ↔ BNHD permute dance to match the kernel's expected layout.

### 4.4 ggml: the `GGML_OP_SPARSE_ATTN` op

Why a new op instead of reusing FlashAttn? Because the access pattern is *gather-then-attend*: each query has its own list of K/V indices (top-k from the indexer). FlashAttn assumes a contiguous KV.

**`ggml/include/ggml.h`** — new enum `GGML_OP_SPARSE_ATTN` (`GGML_OP_COUNT` bumps 96→97), and the factory `ggml_sparse_attn(ctx, q, k, v, mask?, topk, sinks?, scale, max_bias, logit_softcap)`.

**Operand contract (asserted in the factory):**
- `q : [D, n_heads, n_tokens, n_stream]` (F32)
- `k : [D, n_kv,   *,        n_stream]`
- `v : [D, n_kv,   *,        n_stream]`
- `topk: I32 [n_topk, n_tokens, n_heads, n_stream]` — per-(query, head) gather indices.
- `mask: optional F16/F32 [n_topk, n_tokens, 1, n_stream]`
- `sinks: optional F32 [n_heads]`
- output: `[D, n_tokens, n_heads, n_stream]`

**`ggml/src/ggml.c`** — symbol/name tables + the factory. Three `op_params` floats: `scale`, `max_bias`, `logit_softcap`.

**`ggml/src/ggml-cpu/ops.cpp`** — `ggml_compute_forward_sparse_attn_f32`:
- Parallelizes over `(stream, head, query)` triples.
- Per task, gathers `n_topk` rows of K and V into thread-local stack buffers.
- Computes dot products → scale → optional `tanh` softcap → softmax (with optional sink correction `max(max_val, sk[head])` and an extra `exp(sk - max)` term in the denominator) → weighted sum into the output.
- F32-only (the `default` arm `GGML_ABORT`s).

**`ggml/src/ggml-cpu/ggml-cpu.c`** — dispatches the new op, taps the existing `GGML_OP_FLASH_ATTN_EXT` thread-count rule, and reserves zero workspace (`uses thread-local stack allocations`).

**`ggml/src/ggml-backend-meta.cpp`** — registers the op as `handle_generic(scalar_only=true)` for split-tensor backends. This is conservative; non-CPU backends will need a real handler when GPU kernels arrive.

> ⚠️ This commit *defines* `GGML_OP_SPARSE_ATTN` and provides a CPU kernel, but `deepseek4.cpp` does **not** call `build_sparse_attn` yet — the compressed path falls back to `build_attn_mha` over the concatenated KV. The op is groundwork for the indexer-driven top-k selection that Phase 2 will wire up.

### 4.5 Tests

Three test surfaces, registered in `tests/CMakeLists.txt`:

- **`tests/test-deepseek4.cpp` (652 lines)** — round-trip test. It synthesizes a minimal Flash GGUF in-memory (every required tensor with deterministic content), saves it via `llama-model-saver`, reloads, and asserts every expected tensor came back with the right shape and dtype. Use this as the canonical inventory of what a "complete" Flash GGUF looks like.
- **`tests/test-deepseek4-validate.cpp` (467 lines)** — KV-key contract test. Walks the `LLM_KV_*` namespace and asserts every required Flash key is present with the right gguf type. Catches schema drift early.
- **`tests/test-deepseek4-logits.py` (623 lines)** — end-to-end parity test. Invokes the converter on a real HF checkpoint, runs `llama-cli`, runs `transformers`, and compares last-token logits via NMSE. Five prompt cases: `short`, `sliding-window`, `compressed-kv`, `hash-routing`, `mtp-active`. Skip-flags exist so you can reuse a previously converted GGUF.

---

## 5. Onboarding paths by role

### "I want to add another DeepSeek-Vx model"
1. Read `MODEL_TENSORS[DEEPSEEK4]` in `gguf-py/gguf/constants.py` — that is the contract.
2. Read `LLM_ARCH_DEEPSEEK4` cases in `src/llama-model.cpp` (`load_hparams`, `load_tensors`).
3. Read `src/models/deepseek4.cpp` end-to-end; pay attention to which branches are gated by `hc_mult`, `compress_ratio`, `n_hash_layers`, `nextn_predict_layers`.
4. Skim `convert_hf_to_gguf.py:DeepseekV4Model` for the converter idioms (FP8, kv_b_proj skip, eh_proj merge, expert merge).

### "I want to make the sparse-attn path real"
1. Re-read `ggml_sparse_attn` factory and CPU kernel until you can sketch its memory layout.
2. Implement an indexer top-k step in `deepseek4.cpp` for `compress_ratio == 4` layers, producing the `topk: I32 [n_topk, n_tokens, n_heads, n_stream]` operand.
3. Replace the `build_attn_mha(Qcur, k_cat, v_cat, ...)` fallback with `build_sparse_attn(Qcur, k_cat, v_cat, mask, topk, sinks, kq_scale, il)`.
4. Write a CUDA/Metal kernel for `GGML_OP_SPARSE_ATTN` once correctness on CPU is verified.

### "I want to make the FP8 conversion path airtight"
1. Read `dequant_simple` and `_fp8_*` in `convert_hf_to_gguf.py` — note that scale tensors can be `uint8` (e8m0) or already `float`, and weight tensors can be `uint8` (e4m3fn) or already a real torch FP8 dtype.
2. Read `LazyTorchTensor` for the safetensors → torch dtype mapping, especially the `getattr(torch, "float8_e4m3fn", torch.uint8)` fallback.
3. Run `tests/test-deepseek4-logits.py` against a real Flash checkpoint.

### "I just want to read a complete pass through the architecture"
Walk `src/models/deepseek4.cpp` linearly. The constructor is annotated with `cb(cur, "name", il)` callbacks at every step — those names are exactly what shows up in `--verbose` traces, so you can correlate the source with a runtime tensor dump.

---

## 6. Tensor-name cheat-sheet

Block-local, indexed by `bid`:

| GGUF name | Meaning |
|---|---|
| `blk.{bid}.attn_q_a` / `attn_q_b` | LoRA-factored Q (when `q_lora_rank > 0`). |
| `blk.{bid}.attn_kv_a_mqa` | Direct MQA KV projection (`n_embd → head_dim`). |
| `blk.{bid}.attn_o_a` / `attn_o_b` | Grouped low-rank O projection. |
| `blk.{bid}.attn_sinks` | Per-head softmax sink. |
| `blk.{bid}.attn_compressor.{ape,norm,wgate,wkv}` | Compressor for sparse attention. Only when `compress_ratios[bid] > 0`. |
| `blk.{bid}.indexer.attn_k`, `indexer.attn_q_b`, `indexer.proj`, `indexer.k_norm` | Indexer head. Only when `compress_ratios[bid] == 4`. |
| `blk.{bid}.indexer.compressor.{ape,norm,wgate,wkv}` | Indexer's own compressor stream. |
| `blk.{bid}.ffn_gate_tid2eid` | Hash-routing table. Only for `bid < n_hash_layers`. |
| `blk.{bid}.hc_attn_{fn,base,scale}` / `hc_ffn_{fn,base,scale}` | Hyper-connection params per residual site. |
| `blk.{bid}.nextn.eh_proj`, `nextn.enorm`, `nextn.hnorm`, `nextn.shared_head_*`, `nextn.hc_head_*` | MTP block (last `nextn_predict_layers` slots). |

Global:
| GGUF name | Meaning |
|---|---|
| `hc_head_fn`, `hc_head_base`, `hc_head_scale` | Final HC head, applied once before `output_norm`. |

KV-keys worth memorizing:
- `deepseek4.attention.q_lora_rank` / `o_lora_rank` / `output_group_count`
- `deepseek4.hash_layer_count`
- `deepseek4.hyper_connection.{mult,eps,sinkhorn_iters}`
- `deepseek4.attention.layer_compress_ratio` (per-layer array)
- `deepseek4.rope.freq_base_compress`
- `deepseek4.attention.indexer.{head_count,key_length,top_k}`
- `deepseek4.nextn_predict_layers`

---

## 7. Known limitations / `TODO`s explicitly left in this commit

1. **Decode-time compressed KV is not persistent.** The `compress_ratio > 0` path only triggers when `n_tokens >= compress_ratio` (i.e. prefill). Single-token decode falls through to standard attention. Marked `TODO: for decode (n_tokens < compress_ratio), persistent compressed KV state is needed`.
2. **Overlap (`coff == 2`) compressor takes a shortcut.** It views the second half of the pooled output instead of doing a real overlap-and-add. Marked `TODO: implement proper overlap transform`.
3. **No real top-k indexer** for ratio-4 layers. Standard attention is run on the concat of local+compressed KV. Marked `TODO: implement indexer top-k selection for ratio 4 layers`.
4. **`GGML_OP_SPARSE_ATTN` is CPU-only and F32-only.** Other backends fall through `handle_generic(scalar_only=true)`.
5. **Sinkhorn iterations are real but expensive.** `hc_sinkhorn_iters` defaults to 20; for inference you may want to study whether fewer iterations are sufficient.
6. **MTP only chains one block correctly.** The code structurally supports `nextn_predict_layers > 1` (the loop passes `mtp_hidden` to the next iteration), but the reference checkpoint has `num_nextn_predict_layers = 1`, so chained MTP is not exercised.

---

## 8. Suggested study order (one afternoon)

1. **DEEPSEEK_V4_FLASH_FULL_SUPPORT_PLAN.md** (this commit's companion) — five minutes, gives you the phase map.
2. **`gguf-py/gguf/constants.py`** lines added — five minutes, internalize the tensor inventory.
3. **`src/llama-model.cpp` `LLM_ARCH_DEEPSEEK4` block** — fifteen minutes, learn the shapes.
4. **`src/models/deepseek4.cpp`** end-to-end — one hour, the actual algorithm.
5. **`ggml/include/ggml.h` + `ggml/src/ggml-cpu/ops.cpp` `sparse_attn`** — thirty minutes, understand the new op even though it is not yet wired.
6. **`convert_hf_to_gguf.py:DeepseekV4Model`** — thirty minutes, see how HF state-dicts map onto the GGUF schema.
7. **`tests/test-deepseek4.cpp`** — fifteen minutes, treat it as documentation of the expected GGUF.

After this, you are ready to read the full Plan doc and pick a Phase to advance.
