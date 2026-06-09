#include "aceimprovise/inference/stream_inference.hpp"

#include "aceimprovise/stream/pipeline.hpp"

#include "audio-io.h"
#include "model-registry.h"
#include "model-store.h"
#include "philox.h"
#include "pipeline-synth-impl.h"
#include "pipeline-synth-ops.h"
#include "pipeline-synth.h"
#include "task-types.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace aceimprovise {

namespace {

constexpr int kMaxFeedbackDepth = 8;

std::vector<float> blend_latents(const std::vector<float> & a, const std::vector<float> & b, float alpha) {
    const std::size_t n = std::min(a.size(), b.size());
    std::vector<float> out(n);
    const float        w = std::clamp(alpha, 0.0f, 1.0f);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = (1.0f - w) * a[i] + w * b[i];
    }
    return out;
}

}  // namespace

StreamInference::StreamInference() = default;

StreamInference::~StreamInference() {
    dit_holder_.reset();
    if (synth_) {
        ace_synth_free(synth_);
        synth_ = nullptr;
    }
    if (store_) {
        store_free(store_);
        store_ = nullptr;
    }
    if (source_interleaved_) {
        free(source_interleaved_);
        source_interleaved_ = nullptr;
    }
}

bool StreamInference::init(const StreamConfig & config) {
    config_ = config;
    error_.clear();
    session_ = std::make_unique<SynthState>();

    if (config_.models_dir.empty()) {
        error_ = "models_dir not set";
        return false;
    }

    ModelRegistry registry{};
    if (!registry_scan(&registry, config_.models_dir.c_str())) {
        error_ = "no GGUF models found in " + config_.models_dir;
        return false;
    }
    if (registry.dit.empty() || registry.text_enc.empty() || registry.vae.empty()) {
        error_ = "models directory missing DiT, text encoder, or VAE";
        return false;
    }

    store_ = store_create(config_.keep_loaded ? EVICT_NEVER : EVICT_STRICT);
    if (!store_) {
        error_ = "ModelStore creation failed";
        return false;
    }

    if (!ensure_synth()) {
        return false;
    }

    if (!config_.source_path.empty()) {
        int     T_audio = 0;
        float * planar  = audio_read_48k(config_.source_path.c_str(), &T_audio);
        if (!planar || T_audio <= 0) {
            error_ = "failed to load source audio: " + config_.source_path;
            return false;
        }
        source_interleaved_ = audio_planar_to_interleaved(planar, T_audio);
        free(planar);
        source_len_ = T_audio;
    }

    if (!ensure_session_prepared()) {
        return false;
    }

    ready_ = true;
    return true;
}

bool StreamInference::ready() const {
    return ready_;
}

std::string StreamInference::last_error() const {
    return error_;
}

bool StreamInference::ensure_synth() {
    if (synth_) {
        return true;
    }
    ModelRegistry registry{};
    registry_scan(&registry, config_.models_dir.c_str());

    AceSynthParams params{};
    ace_synth_default_params(&params);
    params.text_encoder_path = registry.text_enc[0].path.c_str();
    params.dit_path          = registry.dit[0].path.c_str();
    params.vae_path          = registry.vae[0].path.c_str();
    params.use_fa            = true;
    params.use_batch_cfg     = true;

    synth_ = ace_synth_load(store_, &params);
    if (!synth_) {
        error_ = "ace_synth_load failed";
        return false;
    }
    return true;
}

bool StreamInference::ensure_session_prepared() {
    if (!synth_ || !session_) {
        return false;
    }

    SynthState & s = *session_;
    s.Oc           = synth_->Oc;
    s.ctx_ch       = synth_->ctx_ch;

    const bool have_source = source_interleaved_ != nullptr && source_len_ > 0;
    s.use_source_context   = have_source;
    s.instruction_str      = have_source ? DIT_INSTR_COVER : DIT_INSTR_TEXT2MUSIC;

    if (have_source) {
        if (ops_encode_src(synth_, source_interleaved_, source_len_, nullptr, 0, s) != 0) {
            error_ = "source encode failed";
            return false;
        }
    }

    AceRequest req{};
    request_init(&req);
    req.task_type        = have_source ? TASK_COVER_NOFSQ : TASK_TEXT2MUSIC;
    req.caption          = "ambient electronic";
    req.lyrics           = "[Instrumental]";
    req.inference_steps  = config_.steps;
    req.duration         = have_source ? (float) source_len_ / 48000.0f : 30.0f;
    req.audio_cover_strength = 0.85f;
    req.cover_noise_strength = 0.3f;
    request_resolve_seed(&req);

    if (ops_resolve_params(synth_, &req, 1, s) != 0) {
        error_ = "resolve params failed";
        return false;
    }
    if (ops_resolve_T(synth_, s) != 0) {
        error_ = "resolve T failed";
        return false;
    }
    ops_build_schedule(s);
    if (ops_encode_text(synth_, &req, 1, s) != 0) {
        error_ = "text encode failed";
        return false;
    }
    if (ops_build_context(synth_, &req, 1, s) != 0) {
        error_ = "build context failed";
        return false;
    }
    ops_build_context_silence(synth_, 1, s);

    last_caption_ = req.caption;
    return true;
}

bool StreamInference::refresh_conditioning(const StreamControls & controls) {
    if (!synth_ || !session_) {
        return false;
    }
    if (controls.caption == last_caption_) {
        return true;
    }

    SynthState & s = *session_;
    AceRequest   req{};
    request_init(&req);
    req.task_type           = s.use_source_context ? TASK_COVER_NOFSQ : TASK_TEXT2MUSIC;
    req.caption             = controls.caption.empty() ? "ambient electronic" : controls.caption;
    req.lyrics              = "[Instrumental]";
    req.inference_steps     = config_.steps;
    req.duration            = s.rr.duration > 0 ? s.rr.duration : 30.0f;
    req.audio_cover_strength = controls.cover_strength;
    req.cover_noise_strength = 1.0f - controls.denoise;
    req.guidance_scale      = controls.guidance > 0 ? controls.guidance : 1.0f;
    request_resolve_seed(&req);

    if (ops_encode_text(synth_, &req, 1, s) != 0) {
        error_ = "text re-encode failed";
        return false;
    }
    last_caption_ = req.caption;
    return true;
}

bool StreamInference::ensure_dit_bound() {
    if (!synth_ || !session_) {
        return false;
    }
    if (stepper_.ready()) {
        return true;
    }

    DiTGGML * dit = store_require_dit(synth_->store, synth_->dit_key);
    if (!dit) {
        error_ = "store_require_dit failed";
        return false;
    }
    dit_holder_ = std::make_unique<ModelHandle>(synth_->store, dit);

    SynthState & s = *session_;
    if (!stepper_.bind(dit, config_.max_depth, s.T, s.enc_S, true)) {
        error_ = "DitRingStepper bind failed";
        dit_holder_.reset();
        return false;
    }
    return true;
}

SlotRequest StreamInference::make_submission(const StreamControls & controls) const {
    SlotRequest req;
    req.caption         = controls.caption;
    req.denoise         = controls.denoise;
    req.cover_strength  = controls.cover_strength;
    req.guidance        = controls.guidance;
    req.feedback        = controls.feedback;
    req.feedback_depth  = controls.feedback_depth;
    req.seed            = controls.seed;
    return req;
}

void StreamInference::push_latent_history(const std::vector<float> & latent) {
    latent_history_.push_front(latent);
    while (static_cast<int>(latent_history_.size()) > kMaxFeedbackDepth) {
        latent_history_.pop_back();
    }
}

const std::vector<float> * StreamInference::feedback_latent(int depth) const {
    if (latent_history_.empty()) {
        return nullptr;
    }
    const int idx = std::max(0, std::min(depth - 1, static_cast<int>(latent_history_.size()) - 1));
    return &latent_history_[static_cast<std::size_t>(idx)];
}

std::vector<float> StreamInference::build_evolved_source(const StreamControls & controls) const {
    if (!session_) {
        return {};
    }
    const SynthState & s = *session_;
    const std::vector<float> & base =
        s.noise_blend_latents.empty() ? s.cover_latents : s.noise_blend_latents;
    if (base.empty()) {
        return {};
    }
    const std::vector<float> * fb = feedback_latent(controls.feedback_depth);
    if (controls.feedback > 0.0f && fb != nullptr) {
        return blend_latents(base, *fb, controls.feedback);
    }
    return base;
}

bool StreamInference::init_slot_state(DenoiseSlot & slot, const SlotRequest & req,
                                        const StreamControls & controls) {
    if (!session_) {
        return false;
    }
    SynthState & s = *session_;

    slot.seed           = req.seed;
    slot.denoise        = req.denoise;
    slot.cover_strength = req.cover_strength;
    slot.guidance       = req.guidance;
    slot.switched_cover = false;
    slot.step_idx       = 0;

    const int latent_elems = s.Oc * s.T;
    slot.xt.assign(static_cast<std::size_t>(latent_elems), 0.0f);
    philox_randn(req.seed, slot.xt.data(), latent_elems, true);

    std::vector<float> schedule = s.schedule;
    int                start_idx = 0;
    const float        cover_noise = 1.0f - req.denoise;

    const std::vector<float> evolved = build_evolved_source(controls);
    if (s.use_source_context && s.have_cover && cover_noise > 0.0f && !evolved.empty()) {
        float effective_noise_level = cover_noise;
        float best_dist           = std::fabs(schedule[0] - effective_noise_level);
        for (int i = 1; i < static_cast<int>(schedule.size()); ++i) {
            const float dist = std::fabs(schedule[static_cast<std::size_t>(i)] - effective_noise_level);
            if (dist < best_dist) {
                best_dist = dist;
                start_idx = i;
            }
        }
        const float nearest_t = schedule[static_cast<std::size_t>(start_idx)];
        for (int t = 0; t < s.T; ++t) {
            const int           t_src = t < s.T_cover ? t : s.T_cover - 1;
            const float *       src   = evolved.data() + t_src * s.Oc;
            for (int c = 0; c < s.Oc; ++c) {
                const int idx = t * s.Oc + c;
                slot.xt[static_cast<std::size_t>(idx)] =
                    nearest_t * slot.xt[static_cast<std::size_t>(idx)] + (1.0f - nearest_t) * src[c];
            }
        }
        schedule.erase(schedule.begin(), schedule.begin() + start_idx);
    }

    slot.schedule_offset = start_idx;
    slot.num_steps       = static_cast<int>(schedule.size());
    if (slot.num_steps <= 0) {
        slot.num_steps = static_cast<int>(s.schedule.size());
    }

    // Stash truncated schedule on the slot via num_steps relative to session schedule tail.
    // We index into session_->schedule using schedule_offset + step_idx at tick time.
    return slot.num_steps > 0;
}

bool StreamInference::run_dit_tick(StreamPipeline & pipeline, const StreamControls & controls) {
    if (!ensure_dit_bound() || !session_) {
        return false;
    }

    SynthState & s = *session_;
    const std::vector<StepGroup> groups = pipeline.step_groups();
    if (groups.empty()) {
        return false;
    }

    bool any = false;
    for (const StepGroup & group : groups) {
        const int N = static_cast<int>(group.slot_indices.size());
        if (N <= 0) {
            continue;
        }

        batch_xt_.resize(static_cast<std::size_t>(N * s.Oc * s.T));
        batch_x0_.resize(static_cast<std::size_t>(N * s.Oc * s.T));
        batch_switched_.assign(static_cast<std::size_t>(N), 0);

        for (int i = 0; i < N; ++i) {
            DenoiseSlot * slot = pipeline.slot_at(group.slot_indices[static_cast<std::size_t>(i)]);
            if (!slot) {
                continue;
            }
            memcpy(batch_xt_.data() + static_cast<std::size_t>(i * s.Oc * s.T), slot->xt.data(),
                   static_cast<std::size_t>(s.Oc * s.T) * sizeof(float));
            batch_switched_[static_cast<std::size_t>(i)] = slot->switched_cover ? 1 : 0;
        }

        const int step_idx = group.step_idx;
        const int abs_step = step_idx;  // schedule already truncated per slot at init — use tail schedule
        std::vector<float> sched_tail = s.schedule;
        // Slots may have different schedule offsets; require same offset within group (same denoise at submit).
        DenoiseSlot * ref = pipeline.slot_at(group.slot_indices[0]);
        if (ref && ref->schedule_offset > 0 && ref->schedule_offset < static_cast<int>(sched_tail.size())) {
            sched_tail.erase(sched_tail.begin(), sched_tail.begin() + ref->schedule_offset);
        }
        if (step_idx >= static_cast<int>(sched_tail.size())) {
            continue;
        }

        const int cover_steps =
            ref ? static_cast<int>(static_cast<float>(sched_tail.size()) * ref->cover_strength) : s.cover_steps;

        std::vector<std::uint8_t> switched_flags = batch_switched_;

        DitRingStepParams params{};
        params.batch_n         = N;
        params.step_idx        = step_idx;
        params.num_steps       = static_cast<int>(sched_tail.size());
        params.schedule        = sched_tail.data();
        params.guidance_scale  = controls.guidance > 0 ? controls.guidance : 1.0f;
        params.cover_steps     = cover_steps;
        params.use_batch_cfg   = true;
        params.xt              = batch_xt_.data();
        params.xt_out        = batch_xt_.data();
        params.x0_out          = batch_x0_.data();
        params.context         = s.context.data();
        params.context_switch  = s.context_silence.empty() ? nullptr : s.context_silence.data();
        params.enc_hidden      = s.enc_hidden.data();
        params.enc_switch      = s.enc_hidden_nc.empty() ? nullptr : s.enc_hidden_nc.data();
        params.enc_S           = s.enc_S;
        params.real_S          = s.per_S.data();
        params.real_enc_S      = s.per_enc_S.data();
        params.real_enc_S_switch =
            s.per_enc_S_nc_final.empty() ? nullptr : s.per_enc_S_nc_final.data();
        params.seeds           = s.seeds.data();
        params.switched_cover  = switched_flags.data();

        if (!stepper_.step(params)) {
            error_ = "DiT step failed";
            return false;
        }

        for (int i = 0; i < N; ++i) {
            DenoiseSlot * slot = pipeline.slot_at(group.slot_indices[static_cast<std::size_t>(i)]);
            if (!slot) {
                continue;
            }
            slot->switched_cover = switched_flags[static_cast<std::size_t>(i)] != 0;
            if (step_idx >= slot->num_steps - 1) {
                slot->xt.assign(batch_x0_.begin() + static_cast<std::size_t>(i * s.Oc * s.T),
                                batch_x0_.begin() + static_cast<std::size_t>((i + 1) * s.Oc * s.T));
                slot->step_idx = slot->num_steps;
            } else {
                slot->xt.assign(batch_xt_.begin() + static_cast<std::size_t>(i * s.Oc * s.T),
                                batch_xt_.begin() + static_cast<std::size_t>((i + 1) * s.Oc * s.T));
                pipeline.advance_slot(group.slot_indices[static_cast<std::size_t>(i)]);
            }
            any = true;
        }
    }

    return any;
}

std::vector<std::uint8_t> StreamInference::decode_latent(const std::vector<float> & latent) {
    std::vector<std::uint8_t> out;
    if (!synth_ || !session_ || latent.empty()) {
        return out;
    }

    SynthState decode_state = *session_;
    decode_state.output     = latent;

    AceAudio audio{};
    if (ops_vae_decode(synth_, 1, &audio, decode_state, nullptr, nullptr) != 0 || !audio.samples) {
        error_ = "VAE decode failed";
        return out;
    }

    audio_normalize(audio.samples, audio.n_samples * 2, 10);
    std::string wav = audio_encode_wav(audio.samples, audio.n_samples, audio.sample_rate, WAV_S16);
    ace_audio_free(&audio);
    if (wav.empty()) {
        error_ = "wav encode failed";
        return out;
    }

    out.assign(reinterpret_cast<const std::uint8_t *>(wav.data()),
               reinterpret_cast<const std::uint8_t *>(wav.data()) + wav.size());
    return out;
}

StreamTickResult StreamInference::tick(const StreamControls & controls, StreamPipeline & pipeline) {
    StreamTickResult result;
    if (!ready_) {
        return result;
    }

    const auto t0 = std::chrono::steady_clock::now();

    if (!refresh_conditioning(controls)) {
        return result;
    }

    pipeline.submit(make_submission(controls));
    pipeline.drain_queue([this, &controls](const SlotRequest & req) {
        DenoiseSlot slot;
        init_slot_state(slot, req, controls);
        return slot;
    });

    if (run_dit_tick(pipeline, controls)) {
        result.did_work = true;
    }

    for (std::size_t idx : pipeline.take_finished()) {
        DenoiseSlot * slot = pipeline.slot_at(idx);
        if (!slot) {
            continue;
        }
        push_latent_history(slot->xt);
        result.wav = decode_latent(slot->xt);
        result.finished_slot = idx;
        pipeline.record_finished();
        pipeline.clear_slot(idx);
        result.did_work = true;
        break;  // one decode per wall-clock tick keeps playback paced
    }

    const auto t1 = std::chrono::steady_clock::now();
    result.tick_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    pipeline.tick_mark(result.tick_ms);
    return result;
}

}  // namespace aceimprovise
