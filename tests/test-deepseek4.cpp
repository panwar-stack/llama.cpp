#include "common.h"
#include "log.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "llama-cpp.h"
#include "testing.h"

#include "../src/llama-arch.h"
#include "../src/llama-hparams.h"
#include "../src/llama-model-saver.h"

#include <cinttypes>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <initializer_list>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct tensor_fill_state {
    size_t seed = 0;
    int32_t n_expert = 1;
    std::vector<std::pair<std::string, ggml_type>> seen;
    std::map<std::string, std::array<int64_t, GGML_MAX_DIMS>> dims;

    tensor_fill_state() = default;
    tensor_fill_state(size_t seed, int32_t n_expert) : seed(seed), n_expert(n_expert) {}
};

static bool has_tensor(
        const tensor_fill_state & state,
        const std::string & name,
        ggml_type type = GGML_TYPE_COUNT) {
    for (const auto & item : state.seen) {
        if (item.first == name && (type == GGML_TYPE_COUNT || item.second == type)) {
            return true;
        }
    }
    return false;
}

static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    auto * state = static_cast<tensor_fill_state *>(userdata);
    state->seen.push_back({tensor->name, tensor->type});
    state->dims[tensor->name] = {tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]};

    std::hash<std::string> hasher;
    std::mt19937 gen(hasher(tensor->name) + state->seed);
    std::normal_distribution<float> dis(0.0f, 1.0e-2f);

    const int64_t ne = ggml_nelements(tensor);
    if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            tmp[i] = dis(gen);
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            tmp[i] = ggml_fp32_to_fp16(dis(gen));
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_I32) {
        std::vector<int32_t> tmp(ne);
        if (std::string(tensor->name).find("ffn_gate_tid2eid") != std::string::npos) {
            for (int64_t row = 0; row < tensor->ne[1]; row++) {
                for (int64_t col = 0; col < tensor->ne[0]; col++) {
                    tmp[row * tensor->ne[0] + col] = int32_t((row + col) % state->n_expert);
                }
            }
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ABORT("fatal error");
    }
}

static bool silent_model_load_progress(float /*progress*/, void * /*user_data*/) {
    return true;
}

struct deepseek4_test_spec {
    uint32_t n_vocab = 128;
    uint32_t n_ctx   = 160;
    uint32_t n_embd  = 64;
    uint32_t n_head  = 2;
    uint32_t n_layer = 3;
    uint32_t n_ff    = 96;

    uint32_t q_lora_rank           = 8;
    uint32_t o_lora_rank           = 8;
    uint32_t o_groups              = 2;
    uint32_t n_hash_layers         = 3;
    uint32_t hc_mult               = 0;
    uint32_t nextn_predict_layers  = 0;
    uint32_t n_expert              = 4;
    uint32_t n_expert_used         = 2;
    uint32_t n_expert_shared       = 1;
    uint32_t sliding_window        = 64;
    uint32_t indexer_n_head        = 2;
    uint32_t indexer_head_size     = 16;
    uint32_t indexer_top_k         = 8;

    bool add_tid2eid_metadata = false;

    std::vector<uint32_t> compress_ratios;
};

static void add_tensor_metadata(
        gguf_context * ctx,
        const std::string & name,
        ggml_type type,
        std::initializer_list<int64_t> ne) {
    ggml_tensor t;
    memset(&t, 0, sizeof(t));
    t.type = type;
    size_t i = 0;
    for (const int64_t dim : ne) {
        t.ne[i++] = dim;
    }
    for (; i < GGML_MAX_DIMS; ++i) {
        t.ne[i] = 1;
    }
    t.nb[0] = ggml_type_size(type);
    for (size_t dim = 1; dim < GGML_MAX_DIMS; ++dim) {
        t.nb[dim] = t.ne[dim - 1] * t.nb[dim - 1];
    }
    ggml_set_name(&t, name.c_str());
    gguf_add_tensor(ctx, &t);
}

static gguf_context_ptr create_deepseek4_gguf(const deepseek4_test_spec & spec = {}) {
    gguf_context_ptr ret(gguf_init_empty());
    const llm_arch arch = LLM_ARCH_DEEPSEEK4;
    llama_model_saver ms(arch, ret.get());
    const auto tn = LLM_TN(arch);

    GGML_ASSERT(spec.n_embd % spec.n_head == 0);
    const uint32_t n_embd_head = spec.n_embd / spec.n_head;
    std::vector<uint32_t> compress_ratios = spec.compress_ratios;
    compress_ratios.resize(spec.n_layer, 0);

    // generic required keys
    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,    llm_arch_name(arch));
    ms.add_kv(LLM_KV_VOCAB_SIZE,              spec.n_vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,          spec.n_ctx);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,        spec.n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,             spec.n_layer);
    ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH,     spec.n_ff);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT,    spec.n_head);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, uint32_t(1)); // MQA

    // deepseek4-specific required keys
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,  1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_Q_LORA_RANK,        spec.q_lora_rank);
    ms.add_kv(LLM_KV_ATTENTION_O_LORA_RANK,        spec.o_lora_rank);
    ms.add_kv(LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT, spec.o_groups);
    ms.add_kv(LLM_KV_HASH_LAYER_COUNT,             spec.n_hash_layers);
    ms.add_kv(LLM_KV_HYPER_CONNECTION_MULT,        spec.hc_mult);
    ms.add_kv(LLM_KV_HYPER_CONNECTION_EPS,         1e-6f);
    ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,   spec.n_ff);
    ms.add_kv(LLM_KV_EXPERT_COUNT,                 spec.n_expert);
    ms.add_kv(LLM_KV_EXPERT_USED_COUNT,            spec.n_expert_used);
    ms.add_kv(LLM_KV_EXPERT_SHARED_COUNT,          spec.n_expert_shared);
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW,     spec.sliding_window);
    ms.add_kv(LLM_KV_TOKENIZER_MODEL,              "no_vocab");

    // optional but useful to set explicitly
    ms.add_kv(LLM_KV_LEADING_DENSE_BLOCK_COUNT,       uint32_t(0));
    ms.add_kv(LLM_KV_NEXTN_PREDICT_LAYERS,            spec.nextn_predict_layers);
    ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,            n_embd_head);
    ms.add_kv(LLM_KV_ROPE_FREQ_BASE,                  10000.0f);
    ms.add_kv(LLM_KV_ROPE_FREQ_BASE_COMPRESS,         160000.0f);
    ms.add_kv(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERS, uint32_t(20));
    ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,            1.0f);
    ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,             false);
    ms.add_kv(LLM_KV_EXPERT_GATING_FUNC,              uint32_t(LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS));
    ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIO,        compress_ratios);
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT,    spec.indexer_n_head);
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH,    spec.indexer_head_size);
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K,         spec.indexer_top_k);

    if (spec.add_tid2eid_metadata) {
        const uint32_t hash_layers = std::min(spec.n_hash_layers, spec.n_layer - spec.nextn_predict_layers);
        for (uint32_t il = 0; il < hash_layers; ++il) {
            add_tensor_metadata(ret.get(), tn(LLM_TENSOR_FFN_GATE_TID2EID, "weight", il).str(),
                                GGML_TYPE_I32, {spec.n_expert_used, spec.n_vocab});
        }
    }

    return ret;
}

static std::pair<llama_model_ptr, llama_context_ptr> get_model_and_ctx(
        struct gguf_context * gguf_ctx, tensor_fill_state & state) {
    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = silent_model_load_progress;
    auto * cpu_device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    std::vector<ggml_backend_dev_t> devs = {cpu_device, nullptr};
    model_params.devices = devs.data();
    model_params.split_mode = LLAMA_SPLIT_MODE_LAYER;

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 0;
    ctx_params.n_threads = 4;
    ctx_params.n_threads_batch = 4;
    ctx_params.n_ubatch = 64;

    llama_model_ptr model(llama_model_init_from_user(gguf_ctx, set_tensor_data, &state, model_params));
    if (!model) {
        throw std::runtime_error("failed to create deepseek4 model");
    }
    llama_context_ptr lctx(llama_init_from_model(model.get(), ctx_params));
    if (!lctx) {
        throw std::runtime_error("failed to create llama context");
    }
    return std::make_pair(std::move(model), std::move(lctx));
}

static std::vector<float> get_logits(
        llama_model * model, llama_context * lctx,
        const std::vector<llama_token> & tokens) {
    const uint32_t n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_ctx    = llama_n_ctx(lctx);
    const uint32_t n_tokens = tokens.size();
    llama_batch batch = llama_batch_init(n_ctx, 0, 1);
    GGML_ASSERT(n_tokens <= n_ctx);
    for (uint32_t pos = 0; pos < n_tokens; pos++) {
        common_batch_add(batch, tokens[pos], pos, {0}, true);
    }
    batch.n_tokens = n_tokens;
    if (llama_decode(lctx, batch)) {
        llama_batch_free(batch);
        throw std::runtime_error("failed to decode batch");
    }

    std::vector<float> ret;
    ret.reserve(n_tokens * n_vocab);
    for (uint32_t i = 0; i < n_tokens; i++) {
        const float * logits_ith = llama_get_logits_ith(lctx, i);
        for (uint32_t j = 0; j < n_vocab; j++) {
            ret.push_back(logits_ith[j]);
        }
    }
    llama_batch_free(batch);
    return ret;
}

static std::vector<llama_token> make_tokens(size_t n, uint32_t n_vocab = 128) {
    std::vector<llama_token> tokens;
    tokens.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        tokens.push_back(llama_token(i % n_vocab));
    }
    return tokens;
}

static void assert_logits_sane(testing & t, const std::vector<float> & logits) {
    t.assert_true("logits are produced", !logits.empty());

    bool all_finite = true;
    bool all_zero = true;
    bool all_identical = true;
    const float first = logits.empty() ? 0.0f : logits[0];

    for (float f : logits) {
        all_finite = all_finite && std::isfinite(f);
        all_zero = all_zero && f == 0.0f;
        all_identical = all_identical && f == first;
    }

    t.assert_true("logits are finite", all_finite);
    t.assert_true("logits are not all zero", !all_zero);
    t.assert_true("logits are not all identical", !all_identical);
}

int main() {
    common_init();
    testing t(std::cout);

    const size_t seed = 0x5eed1234;

    t.test("deepseek4", [&](testing & t) {
        t.test("grouped output and hash routing smoke", [&](testing & t) {
            deepseek4_test_spec spec;
            spec.add_tid2eid_metadata = true;

            gguf_context_ptr gguf_ctx = create_deepseek4_gguf(spec);
            tensor_fill_state state{seed, int32_t(spec.n_expert)};
            auto [model, lctx] = get_model_and_ctx(gguf_ctx.get(), state);

            const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
            std::vector<llama_token> tokens = { 0, 1, 7, 63 };
            auto logits = get_logits(model.get(), lctx.get(), tokens);

            t.assert_equal("expected logits size", size_t(tokens.size() * n_vocab), logits.size());
            t.assert_true("tid2eid tensor loaded as I32",
                          has_tensor(state, LLM_TN(LLM_ARCH_DEEPSEEK4)(LLM_TENSOR_FFN_GATE_TID2EID, "weight", 0).str(),
                                     GGML_TYPE_I32));
            t.assert_true("grouped wo_a tensor loaded",
                          has_tensor(state, LLM_TN(LLM_ARCH_DEEPSEEK4)(LLM_TENSOR_ATTN_O_A, "weight", 0).str()));
            t.assert_true("grouped wo_b tensor loaded",
                          has_tensor(state, LLM_TN(LLM_ARCH_DEEPSEEK4)(LLM_TENSOR_ATTN_O_B, "weight", 0).str()));
            assert_logits_sane(t, logits);
        });

        t.test("compressor ratio 4 prefill smoke", [&](testing & t) {
            deepseek4_test_spec spec;
            spec.n_layer = 1;
            spec.n_hash_layers = 0;
            spec.compress_ratios = {4};

            gguf_context_ptr gguf_ctx = create_deepseek4_gguf(spec);
            tensor_fill_state state{seed, int32_t(spec.n_expert)};
            auto [model, lctx] = get_model_and_ctx(gguf_ctx.get(), state);

            std::vector<llama_token> tokens = make_tokens(8, spec.n_vocab);
            auto logits = get_logits(model.get(), lctx.get(), tokens);

            t.assert_true("ratio 4 compressor tensor loaded",
                          has_tensor(state, LLM_TN(LLM_ARCH_DEEPSEEK4)(LLM_TENSOR_ATTN_COMPRESSOR_WKV, "weight", 0).str()));
            assert_logits_sane(t, logits);
        });

        t.test("compressor ratio 128 prefill smoke", [&](testing & t) {
            deepseek4_test_spec spec;
            spec.n_layer = 1;
            spec.n_hash_layers = 0;
            spec.compress_ratios = {128};

            gguf_context_ptr gguf_ctx = create_deepseek4_gguf(spec);
            tensor_fill_state state{seed, int32_t(spec.n_expert)};
            auto [model, lctx] = get_model_and_ctx(gguf_ctx.get(), state);

            std::vector<llama_token> tokens = make_tokens(128, spec.n_vocab);
            auto logits = get_logits(model.get(), lctx.get(), tokens);

            t.assert_true("ratio 128 compressor tensor loaded",
                          has_tensor(state, LLM_TN(LLM_ARCH_DEEPSEEK4)(LLM_TENSOR_ATTN_COMPRESSOR_WKV, "weight", 0).str()));
            assert_logits_sane(t, logits);
        });

        t.test("hyper-connection block smoke", [&](testing & t) {
            deepseek4_test_spec spec;
            spec.n_layer = 1;
            spec.n_hash_layers = 0;
            spec.hc_mult = 1;

            gguf_context_ptr gguf_ctx = create_deepseek4_gguf(spec);
            tensor_fill_state state{seed, int32_t(spec.n_expert)};
            auto [model, lctx] = get_model_and_ctx(gguf_ctx.get(), state);

            std::vector<llama_token> tokens = make_tokens(4, spec.n_vocab);
            auto logits = get_logits(model.get(), lctx.get(), tokens);

            t.assert_true("HC attention tensor loaded",
                          has_tensor(state, LLM_TN(LLM_ARCH_DEEPSEEK4)(LLM_TENSOR_HC_ATTN_FN, "weight", 0).str()));
            t.assert_true("HC output head tensor loaded",
                          has_tensor(state, LLM_TN(LLM_ARCH_DEEPSEEK4)(LLM_TENSOR_HC_HEAD_FN, "weight").str()));
            assert_logits_sane(t, logits);
        });

        t.test("MTP tensor loading and graph smoke", [&](testing & t) {
            deepseek4_test_spec spec;
            spec.n_vocab = 64;
            spec.n_ctx = 64;
            spec.n_embd = 32;
            spec.n_ff = 48;
            spec.q_lora_rank = 4;
            spec.o_lora_rank = 4;
            spec.n_layer = 2;
            spec.n_hash_layers = 0;
            spec.hc_mult = 1;
            spec.nextn_predict_layers = 1;

            gguf_context_ptr gguf_ctx = create_deepseek4_gguf(spec);
            tensor_fill_state state{seed, int32_t(spec.n_expert)};
            auto [model, lctx] = get_model_and_ctx(gguf_ctx.get(), state);

            std::vector<llama_token> tokens = make_tokens(4, spec.n_vocab);
            auto logits = get_logits(model.get(), lctx.get(), tokens);

            t.assert_true("MTP projection tensor loaded",
                          has_tensor(state, LLM_TN(LLM_ARCH_DEEPSEEK4)(LLM_TENSOR_NEXTN_EH_PROJ, "weight", 1).str()));
            t.assert_true("MTP HC head tensor loaded",
                          has_tensor(state, LLM_TN(LLM_ARCH_DEEPSEEK4)(LLM_TENSOR_NEXTN_HC_HEAD_FN, "weight", 1).str()));
            assert_logits_sane(t, logits);
        });

        t.test("deterministic", [&](testing & t) {
            deepseek4_test_spec spec;
            spec.add_tid2eid_metadata = true;

            gguf_context_ptr gguf_ctx1 = create_deepseek4_gguf(spec);
            tensor_fill_state state1{seed, int32_t(spec.n_expert)};
            auto [model1, lctx1] = get_model_and_ctx(gguf_ctx1.get(), state1);
            std::vector<llama_token> tokens = { 0, 1 };
            auto logits1 = get_logits(model1.get(), lctx1.get(), tokens);

            gguf_context_ptr gguf_ctx2 = create_deepseek4_gguf(spec);
            tensor_fill_state state2{seed, int32_t(spec.n_expert)};
            auto [model2, lctx2] = get_model_and_ctx(gguf_ctx2.get(), state2);
            auto logits2 = get_logits(model2.get(), lctx2.get(), tokens);

            t.assert_equal("logits size match", logits1.size(), logits2.size());

            bool all_equal = true;
            for (size_t i = 0; i < logits1.size(); i++) {
                if (logits1[i] != logits2[i]) {
                    all_equal = false;
                    break;
                }
            }
            t.assert_true("logits are deterministic across runs", all_equal);
        });

        t.test("grouped wo_a tensor shape verification", [&](testing & t) {
            deepseek4_test_spec spec;
            spec.n_layer = 1;
            spec.n_hash_layers = 0;
            spec.o_lora_rank = 8;
            spec.o_groups = 2;

            gguf_context_ptr gguf_ctx = create_deepseek4_gguf(spec);
            tensor_fill_state state{seed, int32_t(spec.n_expert)};
            auto [model, lctx] = get_model_and_ctx(gguf_ctx.get(), state);

            const auto tn = LLM_TN(LLM_ARCH_DEEPSEEK4);
            const std::string o_a_name = tn(LLM_TENSOR_ATTN_O_A, "weight", 0).str();

            t.assert_true("attn_o_a tensor exists", has_tensor(state, o_a_name));
            auto it_a = state.dims.find(o_a_name);
            t.assert_true("attn_o_a has 3 dims", it_a != state.dims.end() && it_a->second[2] != 1);
            t.assert_true("attn_o_b tensor exists",
                          has_tensor(state, tn(LLM_TENSOR_ATTN_O_B, "weight", 0).str()));

            auto tokens = make_tokens(1, spec.n_vocab);
            auto logits = get_logits(model.get(), lctx.get(), tokens);
            assert_logits_sane(t, logits);
        });

        t.test("tid2eid tensor shape verification", [&](testing & t) {
            deepseek4_test_spec spec;
            spec.n_layer = 1;
            spec.n_vocab = 128;
            spec.n_hash_layers = 1;
            spec.n_expert_used = 2;
            spec.add_tid2eid_metadata = true;

            gguf_context_ptr gguf_ctx = create_deepseek4_gguf(spec);
            tensor_fill_state state{seed, int32_t(spec.n_expert)};
            auto [model, lctx] = get_model_and_ctx(gguf_ctx.get(), state);

            const auto tn = LLM_TN(LLM_ARCH_DEEPSEEK4);
            const std::string tid2eid_name = tn(LLM_TENSOR_FFN_GATE_TID2EID, "weight", 0).str();

            t.assert_true("tid2eid tensor loaded as I32",
                          has_tensor(state, tid2eid_name, GGML_TYPE_I32));
            auto it = state.dims.find(tid2eid_name);
            t.assert_true("tid2eid ne[0] == n_expert_used",
                          it != state.dims.end() && it->second[0] == int64_t(spec.n_expert_used));
            t.assert_true("tid2eid ne[1] == n_vocab",
                          it != state.dims.end() && it->second[1] == int64_t(spec.n_vocab));

            auto tokens = make_tokens(4, spec.n_vocab);
            auto logits = get_logits(model.get(), lctx.get(), tokens);
            assert_logits_sane(t, logits);
        });

        t.test("compressor ratio 4 indexer tensors present", [&](testing & t) {
            deepseek4_test_spec spec;
            spec.n_layer = 1;
            spec.n_hash_layers = 0;
            spec.compress_ratios = {4};
            spec.indexer_n_head = 2;

            gguf_context_ptr gguf_ctx = create_deepseek4_gguf(spec);
            tensor_fill_state state{seed, int32_t(spec.n_expert)};
            auto [model, lctx] = get_model_and_ctx(gguf_ctx.get(), state);

            const auto tn = LLM_TN(LLM_ARCH_DEEPSEEK4);

            t.assert_true("attn_compressor_wkv loaded",
                          has_tensor(state, tn(LLM_TENSOR_ATTN_COMPRESSOR_WKV, "weight", 0).str()));
            t.assert_true("indexer_compressor_wkv loaded",
                          has_tensor(state, tn(LLM_TENSOR_INDEXER_COMPRESSOR_WKV, "weight", 0).str()));
            t.assert_true("indexer_attn_k loaded",
                          has_tensor(state, tn(LLM_TENSOR_INDEXER_ATTN_K, "weight", 0).str()));
            t.assert_true("indexer_attn_q_b loaded",
                          has_tensor(state, tn(LLM_TENSOR_INDEXER_ATTN_Q_B, "weight", 0).str()));
            t.assert_true("indexer_proj loaded",
                          has_tensor(state, tn(LLM_TENSOR_INDEXER_PROJ, "weight", 0).str()));

            auto tokens = make_tokens(8, spec.n_vocab);
            auto logits = get_logits(model.get(), lctx.get(), tokens);
            assert_logits_sane(t, logits);
        });

        t.test("compressor ratio 128 no indexer tensors", [&](testing & t) {
            deepseek4_test_spec spec;
            spec.n_layer = 1;
            spec.n_hash_layers = 0;
            spec.compress_ratios = {128};

            gguf_context_ptr gguf_ctx = create_deepseek4_gguf(spec);
            tensor_fill_state state{seed, int32_t(spec.n_expert)};
            auto [model, lctx] = get_model_and_ctx(gguf_ctx.get(), state);

            const auto tn = LLM_TN(LLM_ARCH_DEEPSEEK4);

            t.assert_true("attn_compressor_wkv loaded",
                          has_tensor(state, tn(LLM_TENSOR_ATTN_COMPRESSOR_WKV, "weight", 0).str()));
            t.assert_true("indexer_compressor_wkv NOT loaded",
                          !has_tensor(state, tn(LLM_TENSOR_INDEXER_COMPRESSOR_WKV, "weight", 0).str()));
            t.assert_true("indexer_attn_k NOT loaded",
                          !has_tensor(state, tn(LLM_TENSOR_INDEXER_ATTN_K, "weight", 0).str()));

            auto tokens = make_tokens(128, spec.n_vocab);
            auto logits = get_logits(model.get(), lctx.get(), tokens);
            assert_logits_sane(t, logits);
        });

        t.test("sliding window boundary decoding", [&](testing & t) {
            deepseek4_test_spec spec;
            spec.n_layer = 1;
            spec.n_hash_layers = 0;
            spec.sliding_window = 16;
            spec.n_ctx = 64;

            gguf_context_ptr gguf_ctx = create_deepseek4_gguf(spec);
            tensor_fill_state state{seed, int32_t(spec.n_expert)};
            auto [model, lctx] = get_model_and_ctx(gguf_ctx.get(), state);

            auto tokens = make_tokens(32, spec.n_vocab);
            auto logits = get_logits(model.get(), lctx.get(), tokens);
            assert_logits_sane(t, logits);
        });

        t.test("all features combined regression smoke", [&](testing & t) {
            deepseek4_test_spec spec;
            spec.n_layer = 3;
            spec.n_ctx = 512;
            spec.hc_mult = 1;
            spec.compress_ratios = {0, 4, 128};
            spec.n_hash_layers = 1;
            spec.add_tid2eid_metadata = true;
            spec.sliding_window = 64;
            spec.n_expert = 4;
            spec.n_expert_used = 2;
            spec.n_expert_shared = 1;
            spec.o_lora_rank = 8;
            spec.o_groups = 2;
            spec.q_lora_rank = 8;

            gguf_context_ptr gguf_ctx = create_deepseek4_gguf(spec);
            tensor_fill_state state{seed, int32_t(spec.n_expert)};
            auto [model, lctx] = get_model_and_ctx(gguf_ctx.get(), state);

            const auto tn = LLM_TN(LLM_ARCH_DEEPSEEK4);

            // layer 0: hash routing, no compression
            t.assert_true("layer 0 hc_attn_fn loaded",
                          has_tensor(state, tn(LLM_TENSOR_HC_ATTN_FN, "weight", 0).str()));
            t.assert_true("layer 0 hc_ffn_fn loaded",
                          has_tensor(state, tn(LLM_TENSOR_HC_FFN_FN, "weight", 0).str()));
            t.assert_true("layer 0 tid2eid loaded",
                          has_tensor(state, tn(LLM_TENSOR_FFN_GATE_TID2EID, "weight", 0).str(), GGML_TYPE_I32));

            // layer 1: ratio 4 compression with indexer
            t.assert_true("layer 1 hc_attn_fn loaded",
                          has_tensor(state, tn(LLM_TENSOR_HC_ATTN_FN, "weight", 1).str()));
            t.assert_true("layer 1 hc_ffn_fn loaded",
                          has_tensor(state, tn(LLM_TENSOR_HC_FFN_FN, "weight", 1).str()));
            t.assert_true("layer 1 attn_compressor_wkv loaded",
                          has_tensor(state, tn(LLM_TENSOR_ATTN_COMPRESSOR_WKV, "weight", 1).str()));
            t.assert_true("layer 1 indexer_compressor_wkv loaded",
                          has_tensor(state, tn(LLM_TENSOR_INDEXER_COMPRESSOR_WKV, "weight", 1).str()));

            // layer 2: ratio 128 compression, no indexer
            t.assert_true("layer 2 hc_attn_fn loaded",
                          has_tensor(state, tn(LLM_TENSOR_HC_ATTN_FN, "weight", 2).str()));
            t.assert_true("layer 2 hc_ffn_fn loaded",
                          has_tensor(state, tn(LLM_TENSOR_HC_FFN_FN, "weight", 2).str()));
            t.assert_true("layer 2 attn_compressor_wkv loaded",
                          has_tensor(state, tn(LLM_TENSOR_ATTN_COMPRESSOR_WKV, "weight", 2).str()));

            // global HC head tensors
            t.assert_true("hc_head_fn loaded",
                          has_tensor(state, tn(LLM_TENSOR_HC_HEAD_FN, "weight").str()));
            t.assert_true("hc_head_base loaded",
                          has_tensor(state, tn(LLM_TENSOR_HC_HEAD_BASE, "weight").str()));
            t.assert_true("hc_head_scale loaded",
                          has_tensor(state, tn(LLM_TENSOR_HC_HEAD_SCALE, "weight").str()));

            auto tokens = make_tokens(256, spec.n_vocab);
            auto logits = get_logits(model.get(), lctx.get(), tokens);
            assert_logits_sane(t, logits);
        });

        t.test("deterministic with all features", [&](testing & t) {
            deepseek4_test_spec spec;
            spec.n_layer = 3;
            spec.n_ctx = 512;
            spec.hc_mult = 1;
            spec.compress_ratios = {0, 4, 128};
            spec.n_hash_layers = 1;
            spec.add_tid2eid_metadata = true;
            spec.sliding_window = 64;
            spec.n_expert = 4;
            spec.n_expert_used = 2;
            spec.n_expert_shared = 1;
            spec.o_lora_rank = 8;
            spec.o_groups = 2;
            spec.q_lora_rank = 8;

            gguf_context_ptr gguf_ctx1 = create_deepseek4_gguf(spec);
            tensor_fill_state state1{seed, int32_t(spec.n_expert)};
            auto [model1, lctx1] = get_model_and_ctx(gguf_ctx1.get(), state1);
            auto tokens = make_tokens(8, spec.n_vocab);
            auto logits1 = get_logits(model1.get(), lctx1.get(), tokens);

            gguf_context_ptr gguf_ctx2 = create_deepseek4_gguf(spec);
            tensor_fill_state state2{seed, int32_t(spec.n_expert)};
            auto [model2, lctx2] = get_model_and_ctx(gguf_ctx2.get(), state2);
            auto logits2 = get_logits(model2.get(), lctx2.get(), tokens);

            t.assert_equal("logits size match", logits1.size(), logits2.size());

            bool all_equal = true;
            for (size_t i = 0; i < logits1.size(); i++) {
                if (logits1[i] != logits2[i]) {
                    all_equal = false;
                    break;
                }
            }
            t.assert_true("logits are deterministic across runs", all_equal);
        });
    });

    return t.summary();
}
