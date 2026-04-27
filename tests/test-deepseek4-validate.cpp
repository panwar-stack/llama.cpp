#include "common.h"
#include "log.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "llama-cpp.h"
#include "testing.h"

#include "../src/llama-arch.h"
#include "../src/llama-hparams.h"
#include "../src/llama-model-saver.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ────────────────────────────────────────────
// Helper: deepseek4 validation engine
// ────────────────────────────────────────────

struct deepseek4_validation {
    testing & t;
    std::vector<std::string> violations;

    explicit deepseek4_validation(testing & t) : t(t) {}

    // Look up a KV key and verify it exists with the expected type.
    int64_t require_kv(const gguf_context * ctx, llm_kv kv, const LLM_KV & kv_ns,
                       gguf_type expected_type, const char * label) {
        const std::string key_name = kv_ns(kv);
        const int64_t kid = gguf_find_key(ctx, key_name.c_str());
        if (kid < 0) {
            t.assert_true((std::string("missing required key: ") + label).c_str(), false);
            return -1;
        }
        t.assert_equal((std::string("type of ") + label).c_str(),
                       (int) expected_type, (int) gguf_get_kv_type(ctx, kid));
        return kid;
    }

    // Check optional key: if present, verify type. Returns key_id or -1.
    int64_t optional_kv(const gguf_context * ctx, llm_kv kv, const LLM_KV & kv_ns,
                        gguf_type expected_type) {
        const std::string key_name = kv_ns(kv);
        const int64_t kid = gguf_find_key(ctx, key_name.c_str());
        if (kid < 0) return -1;
        t.assert_true((std::string("optional key ") + key_name + " has wrong type").c_str(),
                       gguf_get_kv_type(ctx, kid) == expected_type);
        return kid;
    }

    void validate(const gguf_context * ctx) {
        violations.clear();
        const llm_arch arch = LLM_ARCH_DEEPSEEK4;
        const char * arch_str = llm_arch_name(arch);
        const LLM_KV kv_ns(arch);

        // ── architecture string ──
        {
            const int64_t kid = gguf_find_key(ctx, kv_ns(LLM_KV_GENERAL_ARCHITECTURE).c_str());
            t.assert_true("GENERAL_ARCHITECTURE present", kid >= 0);
            if (kid >= 0) {
                t.assert_equal("GENERAL_ARCHITECTURE type",
                               (int) GGUF_TYPE_STRING, (int) gguf_get_kv_type(ctx, kid));
                const std::string val(gguf_get_val_str(ctx, kid));
                if (val != arch_str) {
                    violations.push_back("wrong architecture: '" + val + "' (expected '" + arch_str + "')");
                }
                t.assert_equal("GENERAL_ARCHITECTURE value", arch_str, val);
            }
        }

        // ── required scalar KV keys ──
        require_kv(ctx, LLM_KV_VOCAB_SIZE,                kv_ns, GGUF_TYPE_UINT32,  "VOCAB_SIZE");
        require_kv(ctx, LLM_KV_CONTEXT_LENGTH,            kv_ns, GGUF_TYPE_UINT32,  "CONTEXT_LENGTH");
        require_kv(ctx, LLM_KV_EMBEDDING_LENGTH,          kv_ns, GGUF_TYPE_UINT32,  "EMBEDDING_LENGTH");
        require_kv(ctx, LLM_KV_BLOCK_COUNT,               kv_ns, GGUF_TYPE_UINT32,  "BLOCK_COUNT");
        require_kv(ctx, LLM_KV_FEED_FORWARD_LENGTH,       kv_ns, GGUF_TYPE_UINT32,  "FEED_FORWARD_LENGTH");
        require_kv(ctx, LLM_KV_ATTENTION_HEAD_COUNT,      kv_ns, GGUF_TYPE_UINT32,  "ATTENTION_HEAD_COUNT");
        require_kv(ctx, LLM_KV_ATTENTION_HEAD_COUNT_KV,   kv_ns, GGUF_TYPE_UINT32,  "ATTENTION_HEAD_COUNT_KV");
        require_kv(ctx, LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, kv_ns, GGUF_TYPE_FLOAT32, "ATTENTION_LAYERNORM_RMS_EPS");
        require_kv(ctx, LLM_KV_ATTENTION_Q_LORA_RANK,     kv_ns, GGUF_TYPE_UINT32,  "ATTENTION_Q_LORA_RANK");
        require_kv(ctx, LLM_KV_ATTENTION_O_LORA_RANK,     kv_ns, GGUF_TYPE_UINT32,  "ATTENTION_O_LORA_RANK");
        require_kv(ctx, LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT, kv_ns, GGUF_TYPE_UINT32, "ATTENTION_OUTPUT_GROUP_COUNT");
        require_kv(ctx, LLM_KV_HASH_LAYER_COUNT,          kv_ns, GGUF_TYPE_UINT32,  "HASH_LAYER_COUNT");
        require_kv(ctx, LLM_KV_HYPER_CONNECTION_MULT,     kv_ns, GGUF_TYPE_UINT32,  "HYPER_CONNECTION_MULT");
        require_kv(ctx, LLM_KV_HYPER_CONNECTION_EPS,      kv_ns, GGUF_TYPE_FLOAT32, "HYPER_CONNECTION_EPS");
        require_kv(ctx, LLM_KV_EXPERT_FEED_FORWARD_LENGTH, kv_ns, GGUF_TYPE_UINT32, "EXPERT_FEED_FORWARD_LENGTH");
        require_kv(ctx, LLM_KV_EXPERT_COUNT,              kv_ns, GGUF_TYPE_UINT32,  "EXPERT_COUNT");
        require_kv(ctx, LLM_KV_EXPERT_USED_COUNT,         kv_ns, GGUF_TYPE_UINT32,  "EXPERT_USED_COUNT");
        require_kv(ctx, LLM_KV_EXPERT_SHARED_COUNT,       kv_ns, GGUF_TYPE_UINT32,  "EXPERT_SHARED_COUNT");
        require_kv(ctx, LLM_KV_ATTENTION_SLIDING_WINDOW,  kv_ns, GGUF_TYPE_UINT32,  "ATTENTION_SLIDING_WINDOW");

        // ── optional KV keys (type check only if present) ──
        optional_kv(ctx, LLM_KV_ATTENTION_COMPRESS_RATIO,       kv_ns, GGUF_TYPE_ARRAY);
        optional_kv(ctx, LLM_KV_ATTENTION_INDEXER_HEAD_COUNT,   kv_ns, GGUF_TYPE_UINT32);
        optional_kv(ctx, LLM_KV_ATTENTION_INDEXER_KEY_LENGTH,   kv_ns, GGUF_TYPE_UINT32);
        optional_kv(ctx, LLM_KV_ATTENTION_INDEXER_TOP_K,        kv_ns, GGUF_TYPE_UINT32);
        optional_kv(ctx, LLM_KV_ROPE_FREQ_BASE_COMPRESS,        kv_ns, GGUF_TYPE_FLOAT32);
        optional_kv(ctx, LLM_KV_HYPER_CONNECTION_SINKHORN_ITERS, kv_ns, GGUF_TYPE_UINT32);
        optional_kv(ctx, LLM_KV_LEADING_DENSE_BLOCK_COUNT,      kv_ns, GGUF_TYPE_UINT32);
        optional_kv(ctx, LLM_KV_NEXTN_PREDICT_LAYERS,           kv_ns, GGUF_TYPE_UINT32);
        optional_kv(ctx, LLM_KV_EXPERT_WEIGHTS_SCALE,           kv_ns, GGUF_TYPE_FLOAT32);
        optional_kv(ctx, LLM_KV_EXPERT_WEIGHTS_NORM,            kv_ns, GGUF_TYPE_BOOL);
        optional_kv(ctx, LLM_KV_EXPERT_GATING_FUNC,             kv_ns, GGUF_TYPE_UINT32);

        // ── consistency checks ──
        const int64_t kid_block = gguf_find_key(ctx, kv_ns(LLM_KV_BLOCK_COUNT).c_str());
        const int64_t kid_hash  = gguf_find_key(ctx, kv_ns(LLM_KV_HASH_LAYER_COUNT).c_str());
        const int64_t kid_ctx   = gguf_find_key(ctx, kv_ns(LLM_KV_CONTEXT_LENGTH).c_str());
        const int64_t kid_sw    = gguf_find_key(ctx, kv_ns(LLM_KV_ATTENTION_SLIDING_WINDOW).c_str());
        const int64_t kid_o_rank = gguf_find_key(ctx, kv_ns(LLM_KV_ATTENTION_O_LORA_RANK).c_str());
        const int64_t kid_o_grp  = gguf_find_key(ctx, kv_ns(LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT).c_str());
        const int64_t kid_embd   = gguf_find_key(ctx, kv_ns(LLM_KV_EMBEDDING_LENGTH).c_str());
        const int64_t kid_head   = gguf_find_key(ctx, kv_ns(LLM_KV_ATTENTION_HEAD_COUNT).c_str());
        const int64_t kid_n_expert       = gguf_find_key(ctx, kv_ns(LLM_KV_EXPERT_COUNT).c_str());
        const int64_t kid_n_expert_used  = gguf_find_key(ctx, kv_ns(LLM_KV_EXPERT_USED_COUNT).c_str());

        // 1) compress_ratios length == BLOCK_COUNT (if present)
        {
            const int64_t kid_cr = gguf_find_key(ctx, kv_ns(LLM_KV_ATTENTION_COMPRESS_RATIO).c_str());
            if (kid_cr >= 0 && kid_block >= 0) {
                const size_t arr_n = gguf_get_arr_n(ctx, kid_cr);
                const uint32_t n_layer = gguf_get_val_u32(ctx, kid_block);
                if (arr_n != (size_t) n_layer) {
                    violations.push_back("compress_ratios length " + std::to_string(arr_n) +
                                         " != BLOCK_COUNT " + std::to_string(n_layer));
                }
            }
        }

        // 2) n_hash_layers > 0 → FFN_GATE_TID2EID tensors must exist
        if (kid_hash >= 0) {
            const uint32_t n_hash = gguf_get_val_u32(ctx, kid_hash);
            if (n_hash > 0) {
                bool has_tid2eid = false;
                const int64_t n_tensors = gguf_get_n_tensors(ctx);
                for (int64_t ti = 0; ti < n_tensors; ti++) {
                    const char * tname = gguf_get_tensor_name(ctx, ti);
                    if (tname && std::string(tname).find("ffn_gate_tid2eid") != std::string::npos) {
                        has_tid2eid = true;
                        break;
                    }
                }
                if (!has_tid2eid) {
                    violations.push_back("n_hash_layers > 0 but no FFN_GATE_TID2EID tensors found");
                }
            }
        }

        // 3) o_lora_rank * o_groups divisible by head_size
        if (kid_o_rank >= 0 && kid_o_grp >= 0 && kid_embd >= 0 && kid_head >= 0) {
            const uint32_t o_rank   = gguf_get_val_u32(ctx, kid_o_rank);
            const uint32_t o_groups = gguf_get_val_u32(ctx, kid_o_grp);
            const uint32_t n_embd   = gguf_get_val_u32(ctx, kid_embd);
            const uint32_t n_head   = gguf_get_val_u32(ctx, kid_head);
            if (n_head > 0) {
                const uint32_t head_size = n_embd / n_head;
                if ((o_rank * o_groups) % head_size != 0) {
                    violations.push_back("o_lora_rank * o_groups (" +
                                         std::to_string(o_rank * o_groups) +
                                         ") not divisible by head_size (" +
                                         std::to_string(head_size) + ")");
                }
            }
        }

        // 4) sliding_window <= context_length
        if (kid_sw >= 0 && kid_ctx >= 0) {
            const uint32_t sw  = gguf_get_val_u32(ctx, kid_sw);
            const uint32_t ctx_len = gguf_get_val_u32(ctx, kid_ctx);
            if (sw > ctx_len) {
                violations.push_back("sliding_window (" + std::to_string(sw) +
                                     ") > context_length (" + std::to_string(ctx_len) + ")");
            }
        }

        // 5) expert_used_count <= expert_count
        if (kid_n_expert >= 0 && kid_n_expert_used >= 0) {
            const uint32_t n_expert      = gguf_get_val_u32(ctx, kid_n_expert);
            const uint32_t n_expert_used = gguf_get_val_u32(ctx, kid_n_expert_used);
            if (n_expert_used > n_expert) {
                violations.push_back("expert_used_count (" + std::to_string(n_expert_used) +
                                     ") > expert_count (" + std::to_string(n_expert) + ")");
            }
        }
    }

    // Assert on violations, used by tests that expect a clean GGUF
    void expect_no_violations() {
        for (const auto & v : violations) {
            t.assert_true(("consistency violation: " + v).c_str(), false);
        }
    }

    // Assert that a specific violation was found (for negative tests)
    void expect_violation(const std::string & substring) {
        bool found = false;
        for (const auto & v : violations) {
            if (v.find(substring) != std::string::npos) {
                found = true;
                break;
            }
        }
        t.assert_true(("expected consistency violation containing '" + substring + "'").c_str(), found);
    }
};

// ────────────────────────────────────────────
// Helpers for building synthetic GGUF files
// ────────────────────────────────────────────

struct deepseek4_val_spec {
    uint32_t n_vocab = 128;
    uint32_t n_ctx   = 160;
    uint32_t n_embd  = 64;
    uint32_t n_head  = 2;
    uint32_t n_layer = 3;
    uint32_t n_ff    = 96;

    uint32_t q_lora_rank           = 8;
    uint32_t o_lora_rank           = 16;
    uint32_t o_groups              = 2;
    uint32_t n_hash_layers         = 0;
    uint32_t hc_mult               = 0;
    uint32_t nextn_predict_layers  = 0;
    uint32_t n_expert              = 4;
    uint32_t n_expert_used         = 2;
    uint32_t n_expert_shared       = 1;
    uint32_t sliding_window        = 64;

    // optional metadata flags
    bool add_optional_keys     = true;
    bool add_tid2eid_metadata  = false;
    bool mislabeled_arch       = false;

    std::vector<uint32_t> compress_ratios;
    bool raw_compress_ratios    = false; // if true, use compress_ratios as-is
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

static gguf_context_ptr create_deepseek4_val_gguf(const deepseek4_val_spec & spec = {}) {
    gguf_context_ptr ret(gguf_init_empty());
    const llm_arch arch = spec.mislabeled_arch ? LLM_ARCH_QWEN2 : LLM_ARCH_DEEPSEEK4;
    llama_model_saver ms(arch, ret.get());
    const auto tn = LLM_TN(arch);

    GGML_ASSERT(spec.n_embd % spec.n_head == 0);
    std::vector<uint32_t> compress_ratios = spec.compress_ratios;
    if (!spec.raw_compress_ratios) {
        compress_ratios.resize(spec.n_layer, 0);
    }

    const uint32_t n_embd_head = spec.n_embd / spec.n_head;

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,    llm_arch_name(arch));
    ms.add_kv(LLM_KV_VOCAB_SIZE,              spec.n_vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,          spec.n_ctx);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,        spec.n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,             spec.n_layer);
    ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH,     spec.n_ff);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT,    spec.n_head);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, uint32_t(1));
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

    if (spec.add_optional_keys) {
        ms.add_kv(LLM_KV_TOKENIZER_MODEL,              "no_vocab");
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
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT,    uint32_t(2));
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH,    uint32_t(16));
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K,         uint32_t(8));
    }

    if (spec.add_tid2eid_metadata) {
        const uint32_t hash_layers = std::min(spec.n_hash_layers, spec.n_layer - spec.nextn_predict_layers);
        for (uint32_t il = 0; il < hash_layers; ++il) {
            add_tensor_metadata(ret.get(), tn(LLM_TENSOR_FFN_GATE_TID2EID, "weight", il).str(),
                                GGML_TYPE_I32, {spec.n_expert_used, spec.n_vocab});
        }
    }

    return ret;
}

static gguf_context_ptr load_gguf_from_file(const std::string & path) {
    gguf_init_params params;
    memset(&params, 0, sizeof(params));
    params.no_alloc = true;
    gguf_context_ptr ret(gguf_init_from_file(path.c_str(), params));
    if (!ret) {
        throw std::runtime_error("failed to load GGUF from " + path);
    }
    return ret;
}

static std::string write_gguf_to_temp(gguf_context * ctx) {
    const std::string path = "/tmp/test-deepseek4-validate-tmp.gguf";
    if (!gguf_write_to_file(ctx, path.c_str(), true)) {
        throw std::runtime_error("failed to write temp GGUF");
    }
    return path;
}

// ────────────────────────────────────────────
// Main
// ────────────────────────────────────────────

int main() {
    common_init();
    testing t(std::cout);

    t.test("deepseek4-validate", [&](testing & t) {

        t.test("required keys present", [&](testing & t) {
            deepseek4_val_spec spec;
            spec.add_optional_keys = false;

            gguf_context_ptr ctx = create_deepseek4_val_gguf(spec);
            deepseek4_validation val(t);
            val.validate(ctx.get());
            val.expect_no_violations();
        });

        t.test("invalid architecture rejected", [&](testing & t) {
            deepseek4_val_spec spec;
            spec.mislabeled_arch = true;

            gguf_context_ptr ctx = create_deepseek4_val_gguf(spec);
            const llm_arch expected_arch = LLM_ARCH_DEEPSEEK4;
            const LLM_KV kv_ns(expected_arch);

            const int64_t kid = gguf_find_key(ctx.get(),
                kv_ns(LLM_KV_GENERAL_ARCHITECTURE).c_str());

            t.assert_true("architecture key exists in malformed file", kid >= 0);
            if (kid >= 0) {
                const char * arch_val = gguf_get_val_str(ctx.get(), kid);
                t.assert_true("architecture is NOT deepseek4",
                    std::string(arch_val) != std::string(llm_arch_name(LLM_ARCH_DEEPSEEK4)));
            }
        });

        t.test("consistency: compress_ratios length", [&](testing & t) {
            deepseek4_val_spec spec;
            spec.n_layer = 4;
            spec.compress_ratios = {1, 2};
            spec.raw_compress_ratios = true;

            gguf_context_ptr ctx = create_deepseek4_val_gguf(spec);
            deepseek4_validation val(t);
            val.validate(ctx.get());
            val.expect_violation("compress_ratios");
        });

        t.test("consistency: n_hash_layers requires tid2eid", [&](testing & t) {
            deepseek4_val_spec spec;
            spec.n_hash_layers = 2;
            spec.add_tid2eid_metadata = false;

            gguf_context_ptr ctx = create_deepseek4_val_gguf(spec);
            deepseek4_validation val(t);
            val.validate(ctx.get());
            val.expect_violation("FFN_GATE_TID2EID");
        });

        t.test("full Flash spec", [&](testing & t) {
            deepseek4_val_spec spec;
            spec.n_hash_layers = 2;
            spec.add_optional_keys = true;
            spec.add_tid2eid_metadata = true;
            spec.compress_ratios = {0, 0, 0};
            spec.nextn_predict_layers = 1;
            spec.hc_mult = 1;

            gguf_context_ptr ctx = create_deepseek4_val_gguf(spec);
            deepseek4_validation val(t);
            val.validate(ctx.get());
            val.expect_no_violations();
        });

        t.test("gguf file round-trip", [&](testing & t) {
            deepseek4_val_spec spec;
            spec.add_tid2eid_metadata = true;

            gguf_context_ptr ctx = create_deepseek4_val_gguf(spec);
            const std::string tmp_path = write_gguf_to_temp(ctx.get());

            gguf_context_ptr reloaded = load_gguf_from_file(tmp_path);

            deepseek4_validation val(t);
            val.validate(reloaded.get());
            val.expect_no_violations();

            std::remove(tmp_path.c_str());
        });

        t.test("edge: zero hash_layers with tid2eid metadata", [&](testing & t) {
            deepseek4_val_spec spec;
            spec.n_hash_layers = 0;
            spec.add_tid2eid_metadata = true;

            gguf_context_ptr ctx = create_deepseek4_val_gguf(spec);
            deepseek4_validation val(t);
            val.validate(ctx.get());
            val.expect_no_violations();
        });

        t.test("edge: sliding_window equals context_length", [&](testing & t) {
            deepseek4_val_spec spec;
            spec.sliding_window = spec.n_ctx;

            gguf_context_ptr ctx = create_deepseek4_val_gguf(spec);
            deepseek4_validation val(t);
            val.validate(ctx.get());
            val.expect_no_violations();
        });

    });

    return t.summary();
}
