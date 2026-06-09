#include "aceimprovise/inference/dit_ring_stepper.hpp"

#include "debug.h"
#include "dit-graph.h"
#include "dit-sampler.h"
#include "dit.h"
#include "solvers/solver-registry.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace aceimprovise {

struct DitRingStepper::Impl {
    DiTGGML * model = nullptr;
    int       max_batch = 0;
    int       T         = 0;
    int       enc_S     = 0;
    int       S         = 0;
    int       Oc        = 0;
    int       ctx_ch    = 0;
    int       in_ch     = 0;
    int       H_enc     = 0;
    int       n_per     = 0;
    bool      use_batch_cfg = true;

    std::vector<uint8_t> ctx_buf;
    ggml_context *       ggml_ctx = nullptr;
    ggml_cgraph *        graph    = nullptr;
    ggml_tensor *        t_input  = nullptr;
    ggml_tensor *        t_output = nullptr;
    ggml_tensor *        t_enc    = nullptr;
    ggml_tensor *        t_t      = nullptr;
    ggml_tensor *        t_tr     = nullptr;
    ggml_tensor *        t_pos    = nullptr;
    ggml_tensor *        t_sa_mask_sw  = nullptr;
    ggml_tensor *        t_sa_mask_pad = nullptr;
    ggml_tensor *        t_ca_mask     = nullptr;

    std::vector<int32_t>  pos_data;
    std::vector<uint16_t> sa_sw_data;
    std::vector<uint16_t> sa_pad_data;
    std::vector<uint16_t> ca_data;

    std::vector<float> input_buf;
    std::vector<float> enc_buf;
    std::vector<float> null_enc_buf;
    std::vector<float> vt;
    std::vector<float> vt_cond;
    std::vector<float> vt_uncond;
    std::vector<APGMomentumBuffer> apg_mbufs;

    bool do_cfg    = false;
    bool batch_cfg = false;
    int  N_graph   = 0;
};

DitRingStepper::DitRingStepper() : impl_(std::make_unique<Impl>()) {}

DitRingStepper::~DitRingStepper() {
    unbind();
}

void DitRingStepper::unbind() {
    if (impl_->ggml_ctx) {
        ggml_free(impl_->ggml_ctx);
        impl_->ggml_ctx = nullptr;
    }
    impl_->graph    = nullptr;
    impl_->model    = nullptr;
    impl_->max_batch = 0;
}

bool DitRingStepper::ready() const {
    return impl_->model != nullptr && impl_->graph != nullptr;
}

bool DitRingStepper::bind(DiTGGML * model, int max_batch, int T, int enc_S, bool use_batch_cfg) {
    unbind();
    if (!model || max_batch < 1 || T <= 0 || enc_S <= 0) {
        return false;
    }

    impl_->model         = model;
    impl_->max_batch     = max_batch;
    impl_->T             = T;
    impl_->enc_S         = enc_S;
    impl_->use_batch_cfg = use_batch_cfg;

    DiTGGMLConfig & c = model->cfg;
    impl_->Oc         = c.out_channels;
    impl_->ctx_ch     = c.in_channels - impl_->Oc;
    impl_->in_ch      = c.in_channels;
    impl_->S          = T / c.patch_size;
    impl_->n_per      = T * impl_->Oc;

    impl_->do_cfg = false;  // resolved per step from guidance_scale
    impl_->batch_cfg = use_batch_cfg;
    impl_->N_graph   = use_batch_cfg ? 2 * max_batch : max_batch;

    size_t ctx_size = ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false);
    impl_->ctx_buf.assign(ctx_size, 0);
    ggml_init_params gparams = {
        /*.mem_size   =*/ctx_size,
        /*.mem_buffer =*/impl_->ctx_buf.data(),
        /*.no_alloc   =*/true,
    };
    impl_->ggml_ctx = ggml_init(gparams);
    impl_->graph =
        dit_ggml_build_graph(model, impl_->ggml_ctx, T, enc_S, impl_->N_graph, &impl_->t_input, &impl_->t_output);

    impl_->t_enc = ggml_graph_get_tensor(impl_->graph, "enc_hidden");
    impl_->H_enc = impl_->t_enc ? (int) impl_->t_enc->ne[0] : 0;
    impl_->t_t   = ggml_graph_get_tensor(impl_->graph, "t");
    impl_->t_tr  = ggml_graph_get_tensor(impl_->graph, "t_r");
    impl_->t_pos = ggml_graph_get_tensor(impl_->graph, "positions");
    impl_->t_sa_mask_sw  = ggml_graph_get_tensor(impl_->graph, "sa_mask_sw");
    impl_->t_sa_mask_pad = ggml_graph_get_tensor(impl_->graph, "sa_mask_pad");
    impl_->t_ca_mask     = ggml_graph_get_tensor(impl_->graph, "ca_mask");

    ggml_backend_sched_reset(model->sched);
    if (model->backend != model->cpu_backend) {
        const char * input_names[] = {"enc_hidden", "input_latents", "t", "t_r", "positions",
                                      "sa_mask_sw", "sa_mask_pad",   "ca_mask"};
        for (const char * iname : input_names) {
            ggml_tensor * t = ggml_graph_get_tensor(impl_->graph, iname);
            if (t) {
                ggml_backend_sched_set_tensor_backend(model->sched, t, model->backend);
            }
        }
    }
    if (!ggml_backend_sched_alloc_graph(model->sched, impl_->graph)) {
        fprintf(stderr, "[DitRingStepper] graph alloc failed\n");
        unbind();
        return false;
    }

    const int N = max_batch;
    const int S = impl_->S;
    const int G = impl_->N_graph;
    impl_->pos_data.resize(S * G);
    for (int b = 0; b < G; ++b) {
        for (int i = 0; i < S; ++i) {
            impl_->pos_data[b * S + i] = i;
        }
    }

    const int win = c.sliding_window;
    impl_->sa_sw_data.assign(S * S * G, 0);
    impl_->sa_pad_data.assign(S * S * G, 0);
    for (int b = 0; b < N; ++b) {
        for (int qi = 0; qi < S; ++qi) {
            for (int ki = 0; ki < S; ++ki) {
                bool real_pos = true;
                int  dist     = (qi > ki) ? (qi - ki) : (ki - qi);
                bool in_win   = (win <= 0) || (S <= win) || (dist <= win);
                int  off      = b * S * S + qi * S + ki;
                impl_->sa_sw_data[off]  = ggml_fp32_to_fp16((real_pos && in_win) ? 0.0f : -INFINITY);
                impl_->sa_pad_data[off] = ggml_fp32_to_fp16(real_pos ? 0.0f : -INFINITY);
            }
        }
    }

    impl_->ca_data.assign(enc_S * S * G, 0);
    for (int b = 0; b < N; ++b) {
        for (int qi = 0; qi < S; ++qi) {
            for (int ki = 0; ki < enc_S; ++ki) {
                impl_->ca_data[b * enc_S * S + qi * enc_S + ki] = ggml_fp32_to_fp16(0.0f);
            }
        }
    }

    impl_->input_buf.assign(impl_->in_ch * T * G, 0.0f);
    impl_->enc_buf.assign(impl_->H_enc * enc_S * G, 0.0f);
    impl_->vt.assign(G * impl_->n_per, 0.0f);

    return true;
}

bool DitRingStepper::step(const DitRingStepParams & p) {
    if (!ready() || p.batch_n < 1 || p.batch_n > impl_->max_batch || !p.schedule || p.num_steps <= 0) {
        return false;
    }
    if (p.step_idx < 0 || p.step_idx >= p.num_steps) {
        return false;
    }

    DiTGGML * model = impl_->model;
    const int N     = p.batch_n;
    const int T     = impl_->T;
    const int Oc    = impl_->Oc;
    const int ctx_ch = impl_->ctx_ch;
    const int in_ch  = impl_->in_ch;
    const int enc_S  = impl_->enc_S;
    const int n_per  = impl_->n_per;
    const int n_total = N * n_per;

    const float guidance = p.guidance_scale;
    impl_->do_cfg        = (guidance > 1.0f) && model->null_condition_emb;
    impl_->batch_cfg     = impl_->do_cfg && p.use_batch_cfg;
    impl_->N_graph       = impl_->batch_cfg ? 2 * N : N;

    if (impl_->do_cfg && (int) impl_->apg_mbufs.size() < N) {
        impl_->apg_mbufs.resize(N);
    }
    if (impl_->do_cfg) {
        if (impl_->batch_cfg) {
            impl_->vt_cond.resize(n_total);
            impl_->vt_uncond.resize(n_total);
            if ((int) impl_->null_enc_buf.size() < impl_->H_enc * enc_S * N) {
                int                emb_n = (int) ggml_nelements(model->null_condition_emb);
                std::vector<float> null_emb(emb_n);
                if (model->null_condition_emb->type == GGML_TYPE_BF16) {
                    std::vector<uint16_t> bf16_buf(emb_n);
                    ggml_backend_tensor_get(model->null_condition_emb, bf16_buf.data(), 0,
                                            emb_n * sizeof(uint16_t));
                    for (int i = 0; i < emb_n; ++i) {
                        uint32_t w = (uint32_t) bf16_buf[i] << 16;
                        memcpy(&null_emb[i], &w, 4);
                    }
                } else {
                    ggml_backend_tensor_get(model->null_condition_emb, null_emb.data(), 0,
                                            emb_n * sizeof(float));
                }
                std::vector<float> null_enc_single(impl_->H_enc * enc_S);
                for (int s = 0; s < enc_S; ++s) {
                    memcpy(&null_enc_single[s * impl_->H_enc], null_emb.data(), impl_->H_enc * sizeof(float));
                }
                impl_->null_enc_buf.resize(impl_->H_enc * enc_S * N);
                for (int b = 0; b < N; ++b) {
                    memcpy(impl_->null_enc_buf.data() + b * enc_S * impl_->H_enc, null_enc_single.data(),
                           enc_S * impl_->H_enc * sizeof(float));
                }
            }
        } else {
            impl_->vt_cond.resize(n_total);
            impl_->vt_uncond.resize(n_total);
        }
    }

    std::vector<float> xt(p.xt, p.xt + n_total);
    float *            xt_io = p.xt_out ? p.xt_out : xt.data();

    // Cover context switch (batch-wide at cover_steps).
    if (p.context_switch && p.cover_steps >= 0 && p.step_idx >= p.cover_steps) {
        for (int b = 0; b < N; ++b) {
            if (p.switched_cover && p.switched_cover[b]) {
                continue;
            }
            for (int t = 0; t < T; ++t) {
                memcpy(&impl_->input_buf[b * T * in_ch + t * in_ch],
                       &p.context_switch[b * T * ctx_ch + t * ctx_ch], ctx_ch * sizeof(float));
            }
            if (p.switched_cover) {
                p.switched_cover[b] = 1;
            }
        }
        if (p.enc_switch) {
            memcpy(impl_->enc_buf.data(), p.enc_switch, impl_->H_enc * enc_S * N * sizeof(float));
            if (p.real_enc_S_switch) {
                for (int b = 0; b < N; ++b) {
                    int re = p.real_enc_S_switch[b];
                    for (int qi = 0; qi < impl_->S; ++qi) {
                        for (int ki = 0; ki < enc_S; ++ki) {
                            float v = (ki < re) ? 0.0f : -INFINITY;
                            impl_->ca_data[b * enc_S * impl_->S + qi * enc_S + ki] = ggml_fp32_to_fp16(v);
                        }
                    }
                }
            }
        }
    }

    const float t_curr = p.schedule[p.step_idx];
    const bool  is_final = (p.step_idx == p.num_steps - 1);

    if (impl_->t_t) {
        ggml_backend_tensor_set(impl_->t_t, &t_curr, 0, sizeof(float));
    }
    if (impl_->t_tr) {
        ggml_backend_tensor_set(impl_->t_tr, &t_curr, 0, sizeof(float));
    }

    // Pack encoder + context for active batch items.
    memcpy(impl_->enc_buf.data(), p.enc_hidden, impl_->H_enc * enc_S * N * sizeof(float));
    if (impl_->do_cfg && impl_->batch_cfg) {
        for (int b = 0; b < N; ++b) {
            memcpy(impl_->enc_buf.data() + (N + b) * enc_S * impl_->H_enc,
                   impl_->null_enc_buf.data() + b * enc_S * impl_->H_enc, enc_S * impl_->H_enc * sizeof(float));
        }
    }

    for (int b = 0; b < N; ++b) {
        for (int t = 0; t < T; ++t) {
            memcpy(&impl_->input_buf[b * T * in_ch + t * in_ch], &p.context[b * T * ctx_ch + t * ctx_ch],
                   ctx_ch * sizeof(float));
        }
        for (int t = 0; t < T; ++t) {
            memcpy(&impl_->input_buf[b * T * in_ch + t * in_ch + ctx_ch], &xt[b * n_per + t * Oc],
                   Oc * sizeof(float));
        }
        if (impl_->batch_cfg) {
            memcpy(&impl_->input_buf[(N + b) * T * in_ch], &impl_->input_buf[b * T * in_ch],
                   T * in_ch * sizeof(float));
        }
    }

    ggml_backend_tensor_set(impl_->t_enc, impl_->enc_buf.data(), 0, impl_->enc_buf.size() * sizeof(float));
    ggml_backend_tensor_set(impl_->t_pos, impl_->pos_data.data(), 0, impl_->S * impl_->N_graph * sizeof(int32_t));
    ggml_backend_tensor_set(impl_->t_sa_mask_sw, impl_->sa_sw_data.data(), 0,
                            impl_->S * impl_->S * impl_->N_graph * sizeof(uint16_t));
    ggml_backend_tensor_set(impl_->t_sa_mask_pad, impl_->sa_pad_data.data(), 0,
                            impl_->S * impl_->S * impl_->N_graph * sizeof(uint16_t));
    ggml_backend_tensor_set(impl_->t_ca_mask, impl_->ca_data.data(), 0,
                            enc_S * impl_->S * impl_->N_graph * sizeof(uint16_t));
    ggml_backend_tensor_set(impl_->t_input, impl_->input_buf.data(), 0, in_ch * T * impl_->N_graph * sizeof(float));

    ggml_backend_sched_graph_compute(model->sched, impl_->graph);

    if (impl_->batch_cfg) {
        std::vector<float> full_output(n_per * impl_->N_graph);
        ggml_backend_tensor_get(impl_->t_output, full_output.data(), 0, n_per * impl_->N_graph * sizeof(float));
        memcpy(impl_->vt_cond.data(), full_output.data(), n_total * sizeof(float));
        memcpy(impl_->vt_uncond.data(), full_output.data() + n_total, n_total * sizeof(float));
        for (int b = 0; b < N; ++b) {
            apg_forward(impl_->vt_cond.data() + b * n_per, impl_->vt_uncond.data() + b * n_per, guidance,
                        impl_->apg_mbufs[b], impl_->vt.data() + b * n_per, Oc, T);
        }
    } else if (impl_->do_cfg) {
        ggml_backend_tensor_get(impl_->t_output, impl_->vt_cond.data(), 0, n_total * sizeof(float));
        ggml_backend_tensor_set(impl_->t_enc, impl_->null_enc_buf.data(), 0, impl_->H_enc * enc_S * N * sizeof(float));
        ggml_backend_tensor_set(impl_->t_input, impl_->input_buf.data(), 0, in_ch * T * N * sizeof(float));
        ggml_backend_sched_graph_compute(model->sched, impl_->graph);
        ggml_backend_tensor_get(impl_->t_output, impl_->vt_uncond.data(), 0, n_total * sizeof(float));
        for (int b = 0; b < N; ++b) {
            apg_forward(impl_->vt_cond.data() + b * n_per, impl_->vt_uncond.data() + b * n_per, guidance,
                        impl_->apg_mbufs[b], impl_->vt.data() + b * n_per, Oc, T);
        }
    } else {
        ggml_backend_tensor_get(impl_->t_output, impl_->vt.data(), 0, n_total * sizeof(float));
    }

    if (is_final) {
        if (p.x0_out) {
            for (int i = 0; i < n_total; ++i) {
                p.x0_out[i] = xt[i] - impl_->vt[i] * t_curr;
            }
        }
    } else {
        const float t_next = p.schedule[p.step_idx + 1];
        const float dt     = t_curr - t_next;
        for (int i = 0; i < n_total; ++i) {
            xt_io[i] = xt[i] - impl_->vt[i] * dt;
        }
    }

    return true;
}

}  // namespace aceimprovise
