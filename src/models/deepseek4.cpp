#include "models.h"
#include "llama-kv-cache.h"

static ggml_tensor * build_hc_pre(
    ggml_context * ctx, ggml_tensor * x,
    ggml_tensor * hc_fn, ggml_tensor * hc_base, ggml_tensor * hc_scale,
    int hc_mult, int sinkhorn_iters, float hc_eps, float norm_eps,
    ggml_tensor *& out_post, ggml_tensor *& out_comb) {

    const int64_t n_embd_head = x->ne[0];
    const int64_t n_tokens    = x->ne[2];
    const int64_t hc_dim      = hc_mult * n_embd_head;
    const int64_t mix_hc      = (2 + hc_mult) * hc_mult;

    float s0 = ((float *)hc_scale->data)[0];
    float s1 = ((float *)hc_scale->data)[1];
    float s2 = ((float *)hc_scale->data)[2];

    ggml_tensor * eps_t      = ggml_fill(ctx, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1), hc_eps);
    ggml_tensor * norm_eps_t = ggml_fill(ctx, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1), norm_eps);
    ggml_tensor * one_t      = ggml_fill(ctx, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1), 1.0f);

    ggml_tensor * x_flat = ggml_reshape_2d(ctx, x, hc_dim, n_tokens);

    ggml_tensor * x_sq   = ggml_sqr(ctx, x_flat);
    ggml_tensor * x_mean = ggml_mean(ctx, x_sq);
    x_mean = ggml_add(ctx, x_mean, norm_eps_t);
    ggml_tensor * x_sqrt = ggml_sqrt(ctx, x_mean);
    ggml_tensor * rsqrt  = ggml_div(ctx, ggml_repeat(ctx, one_t, x_sqrt), x_sqrt);

    ggml_tensor * mixes = ggml_mul_mat(ctx, hc_fn, x_flat);
    mixes = ggml_mul(ctx, mixes, rsqrt);

    // pre: first hc_mult channels, sigmoid + eps
    ggml_tensor * pre_mixes = ggml_view_2d(ctx, mixes, hc_mult, n_tokens,
                                           ggml_row_size(mixes->type, mix_hc), 0);
    pre_mixes = ggml_cont(ctx, pre_mixes);
    ggml_tensor * base_pre  = ggml_view_1d(ctx, hc_base, hc_mult, 0);
    ggml_tensor * pre = ggml_scale(ctx, pre_mixes, s0);
    pre = ggml_add(ctx, pre, base_pre);
    pre = ggml_sigmoid(ctx, pre);
    pre = ggml_add(ctx, pre, eps_t);

    // post: next hc_mult channels, 2 * sigmoid
    ggml_tensor * post_mixes = ggml_view_2d(ctx, mixes, hc_mult, n_tokens,
                                            ggml_row_size(mixes->type, mix_hc),
                                            ggml_row_size(mixes->type, hc_mult));
    post_mixes = ggml_cont(ctx, post_mixes);
    ggml_tensor * base_post  = ggml_view_1d(ctx, hc_base, hc_mult,
                                            ggml_row_size(hc_base->type, hc_mult));
    ggml_tensor * post = ggml_scale(ctx, post_mixes, s1);
    post = ggml_add(ctx, post, base_post);
    post = ggml_sigmoid(ctx, post);
    post = ggml_scale(ctx, post, 2.0f);

    // comb: remaining hc_mult*hc_mult channels, softmax + sinkhorn
    ggml_tensor * comb_mixes = ggml_view_2d(ctx, mixes, hc_mult * hc_mult, n_tokens,
                                            ggml_row_size(mixes->type, mix_hc),
                                            ggml_row_size(mixes->type, 2 * hc_mult));
    comb_mixes = ggml_cont(ctx, comb_mixes);
    ggml_tensor * base_comb  = ggml_view_1d(ctx, hc_base, hc_mult * hc_mult,
                                            ggml_row_size(hc_base->type, 2 * hc_mult));
    ggml_tensor * comb = ggml_scale(ctx, comb_mixes, s2);
    comb = ggml_add(ctx, comb, base_comb);
    comb = ggml_reshape_3d(ctx, comb, hc_mult, hc_mult, n_tokens);

    // row-wise softmax (over cols = ne1 = hc_out) + eps
    {
        ggml_tensor * comb_perm = ggml_permute(ctx, comb, 1, 0, 2, 3);
        comb_perm = ggml_cont(ctx, comb_perm);
        comb_perm = ggml_soft_max(ctx, comb_perm);
        comb_perm = ggml_add(ctx, comb_perm, eps_t);
        comb = ggml_permute(ctx, comb_perm, 1, 0, 2, 3);
    }

    // col normalize: comb = comb / (col_sum + eps), col_sum = sum over ne0 (rows = hc_in)
    {
        ggml_tensor * col_sum     = ggml_sum_rows(ctx, comb);
        ggml_tensor * col_sum_rep = ggml_repeat(ctx, col_sum, comb);
        comb = ggml_div(ctx, comb, ggml_add(ctx, col_sum_rep, eps_t));
    }

    for (int iter = 1; iter < sinkhorn_iters; iter++) {
        // row normalize: comb = comb / (row_sum + eps), row_sum = sum over ne1 (cols = hc_out)
        {
            ggml_tensor * comb_pt     = ggml_permute(ctx, comb, 1, 0, 2, 3);
            comb_pt = ggml_cont(ctx, comb_pt);
            ggml_tensor * row_sum     = ggml_sum_rows(ctx, comb_pt);
            ggml_tensor * row_sum_rep = ggml_repeat(ctx, row_sum, comb_pt);
            comb_pt = ggml_div(ctx, comb_pt, ggml_add(ctx, row_sum_rep, eps_t));
            comb = ggml_permute(ctx, comb_pt, 1, 0, 2, 3);
        }
        // col normalize: comb = comb / (col_sum + eps), col_sum = sum over ne0 (rows = hc_in)
        {
            ggml_tensor * col_sum     = ggml_sum_rows(ctx, comb);
            ggml_tensor * col_sum_rep = ggml_repeat(ctx, col_sum, comb);
            comb = ggml_div(ctx, comb, ggml_add(ctx, col_sum_rep, eps_t));
        }
    }

    out_post = post;
    out_comb = comb;

    // reduce: sum_i pre[i] * x[:, i, :]
    ggml_tensor * x_3d     = ggml_reshape_3d(ctx, x_flat, n_embd_head, hc_mult, n_tokens);
    ggml_tensor * pre_3d   = ggml_reshape_3d(ctx, pre, 1, hc_mult, n_tokens);
    ggml_tensor * weighted = ggml_mul(ctx, x_3d, pre_3d);
    weighted = ggml_permute(ctx, weighted, 1, 0, 2, 3);
    weighted = ggml_cont(ctx, weighted);
    ggml_tensor * result = ggml_sum_rows(ctx, weighted);
    result = ggml_reshape_2d(ctx, result, n_embd_head, n_tokens);

    return result;
}

static ggml_tensor * build_hc_post(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * residual,
    ggml_tensor * post,
    ggml_tensor * comb) {

    const int64_t n_embd_head = x->ne[0];
    const int64_t n_tokens    = x->ne[1];
    const int64_t hc_mult     = residual->ne[1];

    // term1 = post * x
    ggml_tensor * post_3d = ggml_reshape_3d(ctx, post, 1, hc_mult, n_tokens);
    ggml_tensor * x_3d    = ggml_reshape_3d(ctx, x, n_embd_head, 1, n_tokens);
    ggml_tensor * term1   = ggml_mul(ctx, x_3d, post_3d);

    // term2 = sum_i comb[:,i] * residual[:,i,:]
    ggml_tensor * residual_p = ggml_permute(ctx, residual, 1, 0, 2, 3);
    residual_p = ggml_cont(ctx, residual_p);
    ggml_tensor * term2 = ggml_mul_mat(ctx, comb, residual_p);
    term2 = ggml_permute(ctx, term2, 1, 0, 2, 3);

    ggml_tensor * result = ggml_add(ctx, term1, term2);
    return result;
}

static ggml_tensor * build_hc_head(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * hc_head_fn,
    ggml_tensor * hc_head_base,
    ggml_tensor * hc_head_scale,
    int hc_mult, float hc_eps, float norm_eps) {

    const int64_t n_embd_head = x->ne[0];
    const int64_t n_tokens    = x->ne[2];
    const int64_t hc_dim      = hc_mult * n_embd_head;

    float s0 = ((float *)hc_head_scale->data)[0];

    ggml_tensor * eps_t      = ggml_fill(ctx, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1), hc_eps);
    ggml_tensor * norm_eps_t = ggml_fill(ctx, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1), norm_eps);
    ggml_tensor * one_t      = ggml_fill(ctx, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1), 1.0f);

    ggml_tensor * x_flat = ggml_reshape_2d(ctx, x, hc_dim, n_tokens);

    ggml_tensor * x_sq   = ggml_sqr(ctx, x_flat);
    ggml_tensor * x_mean = ggml_mean(ctx, x_sq);
    x_mean = ggml_add(ctx, x_mean, norm_eps_t);
    ggml_tensor * x_sqrt = ggml_sqrt(ctx, x_mean);
    ggml_tensor * rsqrt  = ggml_div(ctx, ggml_repeat(ctx, one_t, x_sqrt), x_sqrt);

    ggml_tensor * mixes = ggml_mul_mat(ctx, hc_head_fn, x_flat);
    mixes = ggml_mul(ctx, mixes, rsqrt);

    ggml_tensor * pre = ggml_scale(ctx, mixes, s0);
    pre = ggml_add(ctx, pre, hc_head_base);
    pre = ggml_sigmoid(ctx, pre);
    pre = ggml_add(ctx, pre, eps_t);

    ggml_tensor * x_3d     = ggml_reshape_3d(ctx, x_flat, n_embd_head, hc_mult, n_tokens);
    ggml_tensor * pre_3d   = ggml_reshape_3d(ctx, pre, 1, hc_mult, n_tokens);
    ggml_tensor * weighted = ggml_mul(ctx, x_3d, pre_3d);
    weighted = ggml_permute(ctx, weighted, 1, 0, 2, 3);
    weighted = ggml_cont(ctx, weighted);
    ggml_tensor * result = ggml_sum_rows(ctx, weighted);
    result = ggml_reshape_2d(ctx, result, n_embd_head, n_tokens);

    return result;
}

llm_build_deepseek4::llm_build_deepseek4(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {

    const int64_t n_embd_head_k = hparams.n_embd_head_k();
    const int64_t n_embd_head_v = hparams.n_embd_head_v();
    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k - n_embd_head_qk_rope;

    const int64_t q_lora_rank  = hparams.n_lora_q;
    const int64_t o_lora_rank  = hparams.n_lora_o;
    const int64_t o_groups     = hparams.n_o_groups;

    const int hc_mult        = (int)hparams.hc_mult;
    const int sinkhorn_iters = (int)hparams.hc_sinkhorn_iters;
    const float hc_eps       = hparams.hc_eps;

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

    // expand to 3D for hyper-connections: {n_embd, hc_mult, n_tokens}
    if (hc_mult > 0) {
        inpL = ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens);
        ggml_tensor * target = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd, hc_mult, n_tokens);
        inpL = ggml_repeat(ctx0, inpL, target);
    }

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn_kv = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = hc_mult == 0 ? build_inp_out_ids() : nullptr;

    int effective_n_layers = hparams.n_layer - hparams.nextn_predict_layers;
    for (int il = 0; il < effective_n_layers; ++il) {
        // After first layer, inpL is 2D from HC post -> expand back to 3D for next layer
        if (hc_mult > 0 && (il > 0 || ggml_n_dims(inpL) < 3)) {
            inpL = ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens);
            ggml_tensor * tgt = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd, hc_mult, n_tokens);
            inpL = ggml_repeat(ctx0, inpL, tgt);
        }
        ggml_tensor * inpSA = inpL;

        // === Attention: norm/hc_pre ===
        ggml_tensor * post_attn = nullptr;
        ggml_tensor * comb_attn = nullptr;

        if (hc_mult > 0) {
            cur = build_hc_pre(ctx0, inpL, model.layers[il].hc_attn_fn, model.layers[il].hc_attn_base,
                              model.layers[il].hc_attn_scale, hc_mult, sinkhorn_iters, hc_eps, norm_rms_eps,
                              post_attn, comb_attn);
            cb(cur, "hc_pre_attn", il);
        } else {
            cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);
        }

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
                weighted_p = ggml_cont(ctx0, weighted_p);
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

            // 4. concatenate local KV + compressed KV along the KV-token axis.
            ggml_tensor * kv_pooled_3d = ggml_reshape_3d(ctx0, kv_pooled, kv_pooled->ne[0], 1, kv_pooled->ne[1]);
            cb(kv_pooled_3d, "kv_pooled_3d", il);

            ggml_tensor * k_cat = ggml_concat(ctx0, k_local, kv_pooled_3d, 2);
            cb(k_cat, "k_cat", il);
            ggml_tensor * v_cat = ggml_concat(ctx0, v_local, kv_pooled_3d, 2);
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

            // Align batch dimensions for grouped matmul:
            //   attn_o_a:  {per_group_dim, o_lora_rank, o_groups}  -> batch=o_groups at ne[2]
            //   o_grouped: {per_group_dim, o_groups,     n_tokens}  -> permute so batch=o_groups at ne[2]
            ggml_tensor * o_grouped_p = ggml_permute(ctx0, o_grouped, 0, 2, 1, 3); // {D, S, G}
            ggml_tensor * o_mid = ggml_mul_mat(ctx0, model.layers[il].attn_o_a, o_grouped_p); // {R, S, G}
            // Permute and reshape to 2D for wo_b projection
            o_mid = ggml_permute(ctx0, o_mid, 2, 0, 1, 3); // {G, R, S}
            o_mid = ggml_cont(ctx0, o_mid);
            cb(o_mid, "attn_o_a", il);

            o_mid = ggml_reshape_2d(ctx0, o_mid, o_groups * o_lora_rank, n_tokens);
            cb(o_mid, "attn_o_a_flat", il);

            cur = ggml_mul_mat(ctx0, model.layers[il].attn_o_b, o_mid);
            cb(cur, "attn_o_b", il);
        } else if (model.layers[il].wo) {
            cur = ggml_mul_mat(ctx0, model.layers[il].wo, cur);
            cb(cur, "attn_wo", il);
        }

        if (il == effective_n_layers - 1 && inp_out_ids && hc_mult == 0) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // === Attention residual: hc_post or ggml_add ===
        ggml_tensor * ffn_inp;
        if (hc_mult > 0) {
            cur = build_hc_post(ctx0, cur, inpSA, post_attn, comb_attn);
            cb(cur, "hc_post_attn", il);
            ffn_inp = cur;
        } else {
            ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);
        }

        // === FFN: norm/hc_pre ===
        ggml_tensor * post_ffn = nullptr;
        ggml_tensor * comb_ffn = nullptr;
        ggml_tensor * res_ffn  = nullptr;

        if (hc_mult > 0) {
            // expand to 3D for HC: {n_embd, hc_mult, n_tokens}
            ggml_tensor * ffn_inp_3d = ggml_reshape_3d(ctx0, ffn_inp, n_embd, 1, n_tokens);
            ggml_tensor * target_ffn = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd, hc_mult, n_tokens);
            ffn_inp_3d = ggml_repeat(ctx0, ffn_inp_3d, target_ffn);
            res_ffn = ffn_inp_3d;

            cur = build_hc_pre(ctx0, ffn_inp_3d, model.layers[il].hc_ffn_fn, model.layers[il].hc_ffn_base,
                              model.layers[il].hc_ffn_scale, hc_mult, sinkhorn_iters, hc_eps, norm_rms_eps,
                              post_ffn, comb_ffn);
            cb(cur, "hc_pre_ffn", il);
        } else {
            cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);
        }

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
            ggml_tensor * moe_out = nullptr;

            if ((uint32_t) il < hparams.n_hash_layers && model.layers[il].ffn_gate_tid2eid && res->t_inp_tokens) {
                // Hash-routed MoE: token-id -> expert-id lookup (DeepSeek-V4 Flash)
                ggml_tensor * selected_experts = ggml_get_rows(ctx0, model.layers[il].ffn_gate_tid2eid, res->t_inp_tokens);
                cb(selected_experts, "ffn_moe_hash_topk", il);

                moe_out = build_moe_ffn(cur,
                    nullptr,
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
                    model.layers[il].ffn_gate_up_exps,
                    nullptr,
                    nullptr,
                    nullptr,
                    selected_experts);
            } else {
                // Score-based routed MoE
                moe_out = build_moe_ffn(cur,
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
            }
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

        // === FFN residual: hc_post or ggml_add ===
        if (hc_mult > 0) {
            cur = build_hc_post(ctx0, cur, res_ffn, post_ffn, comb_ffn);
            cb(cur, "hc_post_ffn", il);
        } else {
            cur = ggml_add(ctx0, cur, ffn_inp);
        }

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    // === Output: hc_head (if HC) then final norm + lm_head ===
    if (hc_mult > 0) {
        cur = build_hc_head(ctx0, cur, model.hc_head_fn, model.hc_head_base, model.hc_head_scale,
                           hc_mult, hc_eps, norm_rms_eps);
        cb(cur, "hc_head", -1);
    }

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = ggml_mul_mat(ctx0, model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);

    // MTP / NextN layers (DeepSeek V4 Flash)
    if (hparams.nextn_predict_layers > 0) {
        // inpL is the HC output of the last main transformer layer
        // Build the MTP path from this point
        ggml_tensor * mtp_inp = inpL;

        for (int im = 0; im < (int)hparams.nextn_predict_layers; im++) {
            int il = effective_n_layers + im;
            auto & layer = model.layers[il];
            auto & nextn = layer.nextn;

            if (!nextn.eh_proj || !nextn.enorm || !nextn.hnorm || !res->t_inp_tokens) {
                break;
            }

            // MTP pre-processing: combine hidden state from previous stage
            // with embedding of current tokens
            // ref: MTPBlock.forward in DeepSeek V4 Flash inference/model.py
            //   e = self.embed(input_ids)
            //   e = self.enorm(e)
            //   x = self.hnorm(x)
            //   x = self.e_proj(e).unsqueeze(2) + self.h_proj(x)

            // 1. Get token embeddings
            ggml_tensor * mtp_emb = ggml_get_rows(ctx0, model.tok_embd, res->t_inp_tokens);
            cb(mtp_emb, "mtp_emb", il);
            mtp_emb = build_norm(mtp_emb, nextn.enorm, NULL, LLM_NORM_RMS, il);
            cb(mtp_emb, "mtp_enorm", il);

            // 2. Apply hnorm to hidden state (HC format: {n_embd, hc_mult, n_tokens})
            ggml_tensor * mtp_hid = build_norm(mtp_inp, nextn.hnorm, NULL, LLM_NORM_RMS, il);
            cb(mtp_hid, "mtp_hnorm", il);

            // 3. Split eh_proj into e_proj (first n_embd rows) and h_proj (last n_embd rows)
            // eh_proj shape: {n_embd, 2 * n_embd}
            const int64_t n_embd_row_bytes = ggml_row_size(nextn.eh_proj->type, n_embd);
            ggml_tensor * e_proj_w = ggml_view_2d(ctx0, nextn.eh_proj, n_embd, n_embd, n_embd_row_bytes, 0);
            cb(e_proj_w, "mtp_e_proj_w", il);
            ggml_tensor * h_proj_w = ggml_view_2d(ctx0, nextn.eh_proj, n_embd, n_embd, n_embd_row_bytes, n_embd * n_embd_row_bytes);
            cb(h_proj_w, "mtp_h_proj_w", il);

            // 4. e_proj(e): project embedding, result {n_embd, n_tokens}
            ggml_tensor * e_out = ggml_mul_mat(ctx0, e_proj_w, mtp_emb);
            cb(e_out, "mtp_e_out", il);

            // 5. h_proj(x): project hidden state, result {n_embd, hc_mult, n_tokens}
            ggml_tensor * h_out = ggml_mul_mat(ctx0, h_proj_w, mtp_hid);
            cb(h_out, "mtp_h_out", il);

            // 6. Broadcast e_out to match h_out shape and add
            // e_out: {n_embd, n_tokens} -> {n_embd, 1, n_tokens}
            ggml_tensor * e_out_3d = ggml_reshape_3d(ctx0, e_out, n_embd, 1, n_tokens);
            cb(e_out_3d, "mtp_e_out_3d", il);
            e_out_3d = ggml_repeat(ctx0, e_out_3d, h_out);
            cb(e_out_3d, "mtp_e_out_broad", il);

            ggml_tensor * mtp_cur = ggml_add(ctx0, e_out_3d, h_out);
            cb(mtp_cur, "mtp_combined", il);

            // 7. Run MTP Block: attention + FFN with HC
            ggml_tensor * mtp_inpSA = mtp_cur;

            // === MTP Attention: hc_pre ===
            ggml_tensor * post_attn = nullptr;
            ggml_tensor * comb_attn = nullptr;

            mtp_cur = build_hc_pre(ctx0, mtp_inpSA, layer.hc_attn_fn, layer.hc_attn_base,
                                   layer.hc_attn_scale, hc_mult, sinkhorn_iters, hc_eps, norm_rms_eps,
                                   post_attn, comb_attn);
            cb(mtp_cur, "mtp_hc_pre_attn", il);

            // MTP self_attention (same pattern as main layers)
            ggml_tensor * mtp_q = NULL;

            if (q_lora_rank > 0) {
                mtp_q = ggml_mul_mat(ctx0, layer.wq_a, mtp_cur);
                cb(mtp_q, "mtp_q", il);
                mtp_q = build_norm(mtp_q, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
                cb(mtp_q, "mtp_q_norm", il);
                mtp_q = ggml_mul_mat(ctx0, layer.wq_b, mtp_q);
                cb(mtp_q, "mtp_q_b", il);
            } else if (layer.wq) {
                mtp_q = ggml_mul_mat(ctx0, layer.wq, mtp_cur);
                cb(mtp_q, "mtp_q", il);
            }

            ggml_tensor * mtp_q_nope =
                ggml_view_3d(ctx0, mtp_q, n_embd_head_qk_nope, n_head, n_tokens,
                             ggml_row_size(mtp_q->type, n_embd_head_k),
                             ggml_row_size(mtp_q->type, n_embd_head_k) * n_head, 0);
            cb(mtp_q_nope, "mtp_q_nope", il);

            ggml_tensor * mtp_q_pe = ggml_view_3d(
                ctx0, mtp_q, n_embd_head_qk_rope, n_head, n_tokens,
                ggml_row_size(mtp_q->type, n_embd_head_k),
                ggml_row_size(mtp_q->type, n_embd_head_k) * n_head,
                ggml_row_size(mtp_q->type, n_embd_head_qk_nope));
            cb(mtp_q_pe, "mtp_q_pe", il);

            // MTP KV projection
            ggml_tensor * mtp_kv = ggml_mul_mat(ctx0, layer.wkv_a_mqa, mtp_cur);
            cb(mtp_kv, "mtp_kv", il);
            mtp_kv = build_norm(mtp_kv, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
            cb(mtp_kv, "mtp_kv_norm", il);

            ggml_tensor * mtp_k_nope = ggml_view_3d(ctx0, mtp_kv, n_embd_head_qk_nope, 1, n_tokens,
                                                     ggml_row_size(mtp_kv->type, n_embd_head_k),
                                                     ggml_row_size(mtp_kv->type, n_embd_head_k), 0);
            cb(mtp_k_nope, "mtp_k_nope", il);

            ggml_tensor * mtp_k_pe = ggml_view_3d(ctx0, mtp_kv, n_embd_head_qk_rope, 1, n_tokens,
                                                   ggml_row_size(mtp_kv->type, n_embd_head_k),
                                                   ggml_row_size(mtp_kv->type, n_embd_head_k),
                                                   ggml_row_size(mtp_kv->type, n_embd_head_qk_nope));
            cb(mtp_k_pe, "mtp_k_pe", il);

            mtp_q_pe = ggml_rope_ext(ctx0, mtp_q_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                                     freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            cb(mtp_q_pe, "mtp_q_pe_rope", il);

            mtp_k_pe = ggml_rope_ext(ctx0, mtp_k_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                                     freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            cb(mtp_k_pe, "mtp_k_pe_rope", il);

            ggml_tensor * mtp_Qcur = ggml_concat(ctx0, mtp_q_nope, mtp_q_pe, 0);
            cb(mtp_Qcur, "mtp_Qcur", il);

            ggml_tensor * mtp_Kcur = ggml_concat(ctx0, mtp_k_nope, mtp_k_pe, 0);
            cb(mtp_Kcur, "mtp_Kcur", il);

            ggml_tensor * mtp_Vcur = ggml_reshape_3d(ctx0, mtp_kv, n_embd_head_k, 1, n_tokens);
            cb(mtp_Vcur, "mtp_Vcur", il);

            // MTP attention: standard path only (MTP layers are decode-time, no compression)
            ggml_tensor * mtp_attn_out = build_attn(inp_attn_kv,
                        nullptr, NULL, nullptr,
                        mtp_Qcur, mtp_Kcur, mtp_Vcur, nullptr,
                        layer.attn_sinks, nullptr, kq_scale, il);
            cb(mtp_attn_out, "mtp_attn_out", il);

            mtp_cur = mtp_attn_out;

            // MTP output projection (grouped low-rank)
            if (layer.attn_o_a && layer.attn_o_b) {
                ggml_tensor * o_grouped = ggml_reshape_3d(ctx0, mtp_cur,
                    n_head * n_embd_head_v / o_groups, o_groups, n_tokens);
                cb(o_grouped, "mtp_o_grouped", il);

                ggml_tensor * o_grouped_p = ggml_permute(ctx0, o_grouped, 0, 2, 1, 3);
                ggml_tensor * o_mid = ggml_mul_mat(ctx0, layer.attn_o_a, o_grouped_p);
                o_mid = ggml_permute(ctx0, o_mid, 2, 0, 1, 3);
                o_mid = ggml_cont(ctx0, o_mid);
                cb(o_mid, "mtp_attn_o_a", il);

                o_mid = ggml_reshape_2d(ctx0, o_mid, o_groups * o_lora_rank, n_tokens);
                cb(o_mid, "mtp_attn_o_a_flat", il);

                mtp_cur = ggml_mul_mat(ctx0, layer.attn_o_b, o_mid);
                cb(mtp_cur, "mtp_attn_o_b", il);
            } else if (layer.wo) {
                mtp_cur = ggml_mul_mat(ctx0, layer.wo, mtp_cur);
                cb(mtp_cur, "mtp_attn_wo", il);
            }

            // MTP Attention residual: hc_post
            ggml_tensor * mtp_ffn_inp;
            mtp_cur = build_hc_post(ctx0, mtp_cur, mtp_inpSA, post_attn, comb_attn);
            cb(mtp_cur, "mtp_hc_post_attn", il);
            mtp_ffn_inp = mtp_cur;

            // === MTP FFN: hc_pre ===
            ggml_tensor * post_ffn = nullptr;
            ggml_tensor * comb_ffn = nullptr;

            mtp_cur = build_hc_pre(ctx0, mtp_ffn_inp, layer.hc_ffn_fn, layer.hc_ffn_base,
                                   layer.hc_ffn_scale, hc_mult, sinkhorn_iters, hc_eps, norm_rms_eps,
                                   post_ffn, comb_ffn);
            cb(mtp_cur, "mtp_hc_pre_ffn", il);

            // MTP FFN: dense or MoE (MTP layers use standard routing, not hash-based)
            if (!layer.ffn_gate_inp) {
                mtp_cur = build_ffn(mtp_cur,
                    layer.ffn_up, NULL, NULL,
                    layer.ffn_gate, NULL, NULL,
                    layer.ffn_down, NULL, NULL,
                    NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(mtp_cur, "mtp_ffn_out", il);
            } else {
                ggml_tensor * moe_out = build_moe_ffn(mtp_cur,
                    layer.ffn_gate_inp,
                    layer.ffn_up_exps,
                    layer.ffn_gate_exps,
                    layer.ffn_down_exps,
                    layer.ffn_exp_probs_b,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, hparams.expert_weights_norm,
                    hparams.expert_weights_scale,
                    (llama_expert_gating_func_type) hparams.expert_gating_func,
                    il,
                    nullptr,
                    layer.ffn_gate_up_exps);
                cb(moe_out, "mtp_ffn_moe_out", il);

                if (layer.ffn_gate_shexp) {
                    ggml_tensor * ffn_shexp =
                        build_ffn(mtp_cur,
                            layer.ffn_up_shexp, NULL, NULL,
                            layer.ffn_gate_shexp, NULL, NULL,
                            layer.ffn_down_shexp, NULL, NULL,
                            NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
                    cb(ffn_shexp, "mtp_ffn_shexp", il);

                    mtp_cur = ggml_add(ctx0, moe_out, ffn_shexp);
                    cb(mtp_cur, "mtp_ffn_out", il);
                } else {
                    mtp_cur = moe_out;
                }
            }

            // MTP FFN residual: hc_post
            mtp_cur = build_hc_post(ctx0, mtp_cur, mtp_ffn_inp, post_ffn, comb_ffn);
            cb(mtp_cur, "mtp_hc_post_ffn", il);

            // === MTP Head: HC head + shared_head_norm + shared_head output ===
            // Save the hidden state (HC format) for potential chained MTP layers
            ggml_tensor * mtp_hidden = mtp_cur;

            if (nextn.hc_head_fn && nextn.hc_head_base && nextn.hc_head_scale) {
                mtp_cur = build_hc_head(ctx0, mtp_cur, nextn.hc_head_fn, nextn.hc_head_base, nextn.hc_head_scale,
                                       hc_mult, hc_eps, norm_rms_eps);
                cb(mtp_cur, "mtp_hc_head", il);
            }

            if (nextn.shared_head_norm) {
                mtp_cur = build_norm(mtp_cur, nextn.shared_head_norm, NULL, LLM_NORM_RMS, il);
                cb(mtp_cur, "mtp_head_norm", il);
            }

            if (nextn.shared_head_head) {
                mtp_cur = ggml_mul_mat(ctx0, nextn.shared_head_head, mtp_cur);
                cb(mtp_cur, "mtp_logits", il);
            }

            // Expand MTP logits into forward graph
            ggml_build_forward_expand(gf, mtp_cur);

            // Pass hidden state (not logits) to next MTP layer if chained
            mtp_inp = mtp_hidden;
        }
    }
}
