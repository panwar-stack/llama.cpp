#include "models.h"
#include "llama-kv-cache.h"

llm_build_deepseek4::llm_build_deepseek4(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {

    const int64_t n_embd_head_k = hparams.n_embd_head_k();
    const int64_t n_embd_head_v = hparams.n_embd_head_v();
    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k - n_embd_head_qk_rope;

    const int64_t q_lora_rank  = hparams.n_lora_q;
    const int64_t o_lora_rank  = hparams.n_lora_o;
    const int64_t o_groups     = hparams.n_o_groups;

    // Pre-scale kq_scale and attn_factor to make the YaRN RoPE work correctly.
    // See https://github.com/ggml-org/llama.cpp/discussions/7416 for detailed explanation.
    GGML_ASSERT(ext_factor >= 0.0f);
    const float attn_factor_org = attn_factor * (1.0f + 0.1f * logf(1.0f / freq_scale));
    const float mscale   = attn_factor_org * (1.0f + 0.1f * hparams.rope_yarn_log_mul * logf(1.0f / freq_scale));
    const float kq_scale = 1.0f * mscale * mscale / sqrtf(float(n_embd_head_k));

    ggml_tensor * cur;
    ggml_tensor * inpL;

    // {n_embd, n_tokens}
    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn_kv = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    int effective_n_layers = hparams.n_layer - hparams.nextn_predict_layers;
    for (int il = 0; il < effective_n_layers; ++il) {
        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self_attention
        ggml_tensor * q = NULL;

        if (q_lora_rank > 0) {
            q = ggml_mul_mat(ctx0, model.layers[il].wq_a, cur);
            cb(q, "q", il);

            q = build_norm(q, model.layers[il].attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
            cb(q, "q_norm", il);

            q = ggml_mul_mat(ctx0, model.layers[il].wq_b, q);
            cb(q, "q", il);
        } else {
            q = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
            cb(q, "q", il);
        }

        // split into {n_embd_head_qk_nope, n_head, n_tokens}
        ggml_tensor * q_nope =
            ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head, n_tokens, ggml_row_size(q->type, n_embd_head_k),
                         ggml_row_size(q->type, n_embd_head_k) * n_head, 0);
        cb(q_nope, "q_nope", il);

        // and {n_embd_head_qk_rope, n_head, n_tokens}
        ggml_tensor * q_pe = ggml_view_3d(
            ctx0, q, n_embd_head_qk_rope, n_head, n_tokens, ggml_row_size(q->type, n_embd_head_k),
            ggml_row_size(q->type, n_embd_head_k) * n_head, ggml_row_size(q->type, n_embd_head_qk_nope));
        cb(q_pe, "q_pe", il);

        // V4: direct MQA KV projection to head_dim (no MLA)
        ggml_tensor * kv = ggml_mul_mat(ctx0, model.layers[il].wkv_a_mqa, cur);
        cb(kv, "kv", il);

        kv = build_norm(kv, model.layers[il].attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
        cb(kv, "kv_norm", il);

        // split KV into nope and rope portions
        ggml_tensor * k_nope = ggml_view_3d(ctx0, kv, n_embd_head_qk_nope, 1, n_tokens,
                                            ggml_row_size(kv->type, n_embd_head_k),
                                            ggml_row_size(kv->type, n_embd_head_k), 0);
        cb(k_nope, "k_nope", il);

        ggml_tensor * k_pe = ggml_view_3d(ctx0, kv, n_embd_head_qk_rope, 1, n_tokens,
                                          ggml_row_size(kv->type, n_embd_head_k),
                                          ggml_row_size(kv->type, n_embd_head_k),
                                          ggml_row_size(kv->type, n_embd_head_qk_nope));
        cb(k_pe, "k_pe", il);

        q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                             ext_factor, attn_factor, beta_fast, beta_slow);
        cb(q_pe, "q_pe", il);

        k_pe = ggml_rope_ext(ctx0, k_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                             ext_factor, attn_factor, beta_fast, beta_slow);
        cb(k_pe, "k_pe", il);

        ggml_tensor * Qcur = ggml_concat(ctx0, q_nope, q_pe, 0);
        cb(Qcur, "Qcur", il);

        // V4: MQA - k_nope and k_pe both have ne1=1, no need to repeat
        ggml_tensor * Kcur = ggml_concat(ctx0, k_nope, k_pe, 0);
        cb(Kcur, "Kcur", il);

        ggml_tensor * Vcur = ggml_reshape_3d(ctx0, kv, n_embd_head_k, 1, n_tokens);
        cb(Vcur, "Vcur", il);

        ggml_tensor * attn_out = nullptr;

        const uint32_t compress_ratio = hparams.compress_ratios[il];
        GGML_UNUSED(hparams.n_swa); // sliding window is implicit in the KV cache / topk indices

        if (compress_ratio > 0 && n_tokens >= (int64_t) compress_ratio) {
            // DeepSeek V4 Flash compressed sparse attention path
            // TODO: for decode (n_tokens < compress_ratio), persistent compressed KV state is needed

            // 1. store local KV in standard cache
            const auto & k_idxs = inp_attn_kv->get_k_idxs();
            const auto & v_idxs = inp_attn_kv->get_v_idxs();
            ggml_build_forward_expand(gf, inp_attn_kv->mctx->cpy_k(ctx0, Kcur, k_idxs, il));
            ggml_build_forward_expand(gf, inp_attn_kv->mctx->cpy_v(ctx0, Vcur, v_idxs, il));

            // 2. use current batch KV as local KV (prefill path)
            ggml_tensor * k_local = Kcur;
            ggml_tensor * v_local = Vcur;

            // 3. compute compressed KV from attention input (cur)
            // compressor projects cur to [coff * head_dim, n_tokens]
            const int64_t coff = compress_ratio == 4 ? 2 : 1;

            ggml_tensor * kv_comp = ggml_mul_mat(ctx0, model.layers[il].attn_compressor_wkv, cur);
            cb(kv_comp, "kv_comp", il);

            ggml_tensor * score = ggml_mul_mat(ctx0, model.layers[il].attn_compressor_wgate, cur);
            cb(score, "comp_score", il);

            // add absolute position embedding
            // ape shape: [coff * head_dim, compress_ratio]
            // we need to tile it to [coff * head_dim, n_tokens]
            ggml_tensor * ape = model.layers[il].attn_compressor_ape;
            if (ape) {
                ggml_tensor * ape_tiled = ggml_repeat(ctx0, ape, kv_comp);
                cb(ape_tiled, "ape_tiled", il);
                score = ggml_add(ctx0, score, ape_tiled);
                cb(score, "comp_score_ape", il);
            }

            // reshape to [head_dim, coff, compress_ratio, n_tokens/compress_ratio] or similar
            // For simplicity, do non-overlapping pooling: group tokens into chunks of size compress_ratio
            ggml_tensor * kv_pooled = nullptr;
            {
                // kv_comp shape: [coff * head_dim, n_tokens]
                // Reshape to [coff * head_dim, compress_ratio, n_tokens / compress_ratio]
                const int64_t n_chunks = n_tokens / compress_ratio;
                ggml_tensor * kv_r = ggml_reshape_3d(ctx0, kv_comp, coff * n_embd_head_k, compress_ratio, n_chunks);
                cb(kv_r, "kv_r", il);

                ggml_tensor * score_r = ggml_reshape_3d(ctx0, score, coff * n_embd_head_k, compress_ratio, n_chunks);
                cb(score_r, "score_r", il);

                // softmax over compress_ratio dimension (dim 1)
                score_r = ggml_soft_max(ctx0, score_r);
                cb(score_r, "score_sm", il);

                // weighted sum: sum over compress_ratio dimension
                ggml_tensor * weighted = ggml_mul(ctx0, kv_r, score_r);
                cb(weighted, "weighted", il);

                // permute so compress_ratio is dim 0, then sum_rows
                ggml_tensor * weighted_p = ggml_permute(ctx0, weighted, 1, 0, 2, 3);
                cb(weighted_p, "weighted_p", il);

                kv_pooled = ggml_sum_rows(ctx0, weighted_p);
                cb(kv_pooled, "kv_pooled", il);

                // reshape back to [coff * head_dim, n_chunks]
                kv_pooled = ggml_reshape_2d(ctx0, kv_pooled, coff * n_embd_head_k, n_chunks);
                cb(kv_pooled, "kv_pooled_2d", il);

                // kv_pooled is now [coff * head_dim, 1, n_chunks] -> need to collapse to [head_dim, n_chunks] for overlap
                if (coff == 2) {
                    // overlap mode: interleave overlapping and non-overlapping parts
                    // For simplicity, just take the second half (normal compression)
                    // TODO: implement proper overlap transform
                    kv_pooled = ggml_view_2d(ctx0, kv_pooled, n_embd_head_k, n_chunks,
                                             ggml_row_size(kv_pooled->type, coff * n_embd_head_k),
                                             ggml_row_size(kv_pooled->type, n_embd_head_k));
                    cb(kv_pooled, "kv_pooled_half", il);
                } else {
                    kv_pooled = ggml_reshape_2d(ctx0, kv_pooled, n_embd_head_k, n_chunks);
                    cb(kv_pooled, "kv_pooled_flat", il);
                }
            }

            // normalize
            kv_pooled = build_norm(kv_pooled, model.layers[il].attn_compressor_norm, nullptr, LLM_NORM_RMS, il);
            cb(kv_pooled, "kv_pooled_norm", il);

            // apply compressed RoPE
            if (n_embd_head_qk_rope > 0) {
                ggml_tensor * k_pooled_pe = ggml_view_3d(ctx0, kv_pooled, n_embd_head_qk_rope, 1, kv_pooled->ne[1],
                                                         ggml_row_size(kv_pooled->type, n_embd_head_k),
                                                         ggml_row_size(kv_pooled->type, n_embd_head_k),
                                                         ggml_row_size(kv_pooled->type, n_embd_head_qk_nope));
                cb(k_pooled_pe, "k_pooled_pe", il);

                // positions for compressed tokens: every compress_ratio-th token
                ggml_tensor * comp_pos_f = ggml_arange(ctx0, (float)(compress_ratio - 1), (float)n_tokens, (float)compress_ratio);
                cb(comp_pos_f, "comp_pos_f", il);
                ggml_tensor * comp_pos = ggml_cast(ctx0, comp_pos_f, GGML_TYPE_I32);
                cb(comp_pos, "comp_pos", il);

                k_pooled_pe = ggml_rope_ext(ctx0, k_pooled_pe, comp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                                             hparams.rope_freq_base_compress, freq_scale,
                                             ext_factor, attn_factor, beta_fast, beta_slow);
                cb(k_pooled_pe, "k_pooled_pe_rope", il);
            }

            // 4. concatenate local KV + compressed KV
            // Need to match dimensions: k_local is [head_dim, n_tokens, 1, 1] in implicit terms
            // kv_pooled is [head_dim, n_chunks]. Expand to match.
            ggml_tensor * kv_pooled_4d = ggml_reshape_4d(ctx0, kv_pooled, kv_pooled->ne[0], kv_pooled->ne[1], 1, 1);
            cb(kv_pooled_4d, "kv_pooled_4d", il);

            ggml_tensor * k_cat = ggml_concat(ctx0, k_local, kv_pooled_4d, 1);
            cb(k_cat, "k_cat", il);
            ggml_tensor * v_cat = ggml_concat(ctx0, v_local, kv_pooled_4d, 1);
            cb(v_cat, "v_cat", il);

            // 5. build topk indices (simplified: attend to all local + all compressed)
            // For now, just use standard attention on the concatenated KV
            // TODO: implement indexer top-k selection for ratio 4 layers
            const auto & kq_mask = inp_attn_kv->get_kq_mask();

            attn_out = build_attn_mha(Qcur, k_cat, v_cat, nullptr, kq_mask,
                                      model.layers[il].attn_sinks, nullptr, kq_scale, il);
            cb(attn_out, "attn_out_sparse", il);
        } else {
            // Standard attention path (pure SWA or decode without compression)
            attn_out = build_attn(inp_attn_kv,
                        nullptr, NULL, nullptr,
                        Qcur, Kcur, Vcur, nullptr,
                        model.layers[il].attn_sinks, nullptr, kq_scale, il);
            cb(attn_out, "attn_out", il);
        }

        cur = attn_out;

        // V4: low-rank grouped output projection
        if (model.layers[il].attn_o_a && model.layers[il].attn_o_b) {
            // Reshape attention output from [n_head*head_dim, n_tokens] to [n_head*head_dim/o_groups, o_groups, n_tokens]
            ggml_tensor * o_grouped = ggml_reshape_3d(ctx0, cur, n_head * n_embd_head_v / o_groups, o_groups, n_tokens);
            cb(o_grouped, "o_grouped", il);

            // Apply the per-group low-rank projection from the reference implementation:
            // o = einsum("bsgd,grd->bsgr", o, wo_a); x = wo_b(o.flatten(2))
            ggml_tensor * o_mid = ggml_mul_mat(ctx0, model.layers[il].attn_o_a, o_grouped);
            cb(o_mid, "attn_o_a", il);

            o_mid = ggml_reshape_2d(ctx0, o_mid, o_groups * o_lora_rank, n_tokens);
            cb(o_mid, "attn_o_a_flat", il);

            cur = ggml_mul_mat(ctx0, model.layers[il].attn_o_b, o_mid);
            cb(cur, "attn_o_b", il);
        } else if (model.layers[il].wo) {
            cur = ggml_mul_mat(ctx0, model.layers[il].wo, cur);
            cb(cur, "attn_wo", il);
        }

        if (il == effective_n_layers - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if ((uint32_t) il < hparams.n_layer_dense_lead || !model.layers[il].ffn_gate_inp) {
            // Dense FFN
            cur = build_ffn(cur,
                model.layers[il].ffn_up, NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            // MoE branch
            ggml_tensor * moe_out = build_moe_ffn(cur,
                model.layers[il].ffn_gate_inp,
                model.layers[il].ffn_up_exps,
                model.layers[il].ffn_gate_exps,
                model.layers[il].ffn_down_exps,
                model.layers[il].ffn_exp_probs_b,
                n_expert, n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il,
                nullptr,
                model.layers[il].ffn_gate_up_exps);
            cb(moe_out, "ffn_moe_out", il);

            // FFN shared expert
            {
                ggml_tensor * ffn_shexp =
                    build_ffn(cur,
                        model.layers[il].ffn_up_shexp, NULL, NULL,
                        model.layers[il].ffn_gate_shexp, NULL, NULL,
                        model.layers[il].ffn_down_shexp, NULL, NULL,
                        NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(ffn_shexp, "ffn_shexp", il);

                cur = ggml_add(ctx0, moe_out, ffn_shexp);
                cb(cur, "ffn_out", il);
            }
        }
        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = ggml_mul_mat(ctx0, model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
