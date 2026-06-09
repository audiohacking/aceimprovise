#include "aceimprovise/session/performance_session.hpp"

#include "aceimprovise/liveset/compositor.hpp"

#include <chrono>

namespace aceimprovise {

PerformanceSession::PerformanceSession(SessionConfig config)
    : config_(std::move(config)), pipeline_(config_.pipeline_depth, config_.steps) {
    StreamConfig infer_cfg;
    infer_cfg.models_dir  = config_.models_dir;
    infer_cfg.source_path = config_.source_path;
    infer_cfg.keep_loaded = config_.keep_loaded;
    infer_cfg.steps       = config_.steps;
    infer_cfg.max_depth   = config_.pipeline_depth;

    if (!infer_cfg.models_dir.empty()) {
        if (!inference_.init(infer_cfg)) {
            fprintf(stderr, "[aceimprovise] Stream inference init: %s\n", inference_.last_error().c_str());
        }
    }
}

PerformanceSession::~PerformanceSession() {
    stop();
}

void PerformanceSession::set_audio_callback(AudioCallback cb) {
    audio_callback_ = std::move(cb);
}

void PerformanceSession::set_state_callback(StateCallback cb) {
    state_callback_ = std::move(cb);
}

void PerformanceSession::bump_generation_if_playing() {
    if (playing_) {
        seed_counter_.fetch_add(1);
        if (state_callback_) {
            state_callback_();
        }
    }
}

Liveset & PerformanceSession::liveset() {
    return liveset_;
}

const Liveset & PerformanceSession::liveset() const {
    return liveset_;
}

PromptEntry PerformanceSession::prompt_add(const std::string & caption, float weight) {
    PromptEntry entry = liveset_.add(caption, weight);
    if (inference_.ready()) {
        liveset_.mark_encode_ready(entry.id);
    } else {
        simulate_encode(entry);
    }
    bump_generation_if_playing();
    return entry;
}

bool PerformanceSession::prompt_remove(const std::string & id) {
    const bool ok = liveset_.remove(id);
    if (ok) {
        bump_generation_if_playing();
    }
    return ok;
}

bool PerformanceSession::prompt_modify(const std::string & id, const std::string & caption) {
    const bool ok = liveset_.modify(id, caption);
    if (ok && !inference_.ready()) {
        if (auto entry = liveset_.get(id)) {
            simulate_encode(*entry);
        }
    } else if (ok && inference_.ready()) {
        liveset_.mark_encode_ready(id);
    }
    if (ok) {
        bump_generation_if_playing();
    }
    return ok;
}

bool PerformanceSession::prompt_weight(const std::string & id, float weight) {
    const bool ok = liveset_.set_weight(id, weight);
    if (ok) {
        bump_generation_if_playing();
    }
    return ok;
}

bool PerformanceSession::prompt_mute(const std::string & id, bool muted) {
    const bool ok = liveset_.set_muted(id, muted);
    if (ok) {
        bump_generation_if_playing();
    }
    return ok;
}

void PerformanceSession::set_shared_denoise(float v) {
    shared_.set_denoise(v);
    bump_generation_if_playing();
}

void PerformanceSession::set_shared_cover(float v) {
    shared_.set_cover_strength(v);
    bump_generation_if_playing();
}

void PerformanceSession::set_shared_guidance(float v) {
    shared_.set_guidance(v);
    bump_generation_if_playing();
}

void PerformanceSession::set_shared_feedback(float v) {
    shared_.set_feedback(v);
    bump_generation_if_playing();
}

void PerformanceSession::set_shared_feedback_depth(int v) {
    shared_.set_feedback_depth(v);
    bump_generation_if_playing();
}

void PerformanceSession::start() {
    if (playing_.exchange(true)) {
        return;
    }
    stop_requested_ = false;
    seed_counter_.fetch_add(1);

    if (inference_.ready()) {
        worker_ = std::thread([this] { stream_loop(); });
    } else {
        worker_ = std::thread([this] {
            using namespace std::chrono_literals;
            while (!stop_requested_) {
                pipeline_.tick_mark(1.0);
                std::this_thread::sleep_for(40ms);
            }
        });
    }
}

void PerformanceSession::stop() {
    if (!playing_.exchange(false)) {
        return;
    }
    stop_requested_ = true;
    if (worker_.joinable()) {
        worker_.join();
    }
    generating_ = false;
    pipeline_.flush();
}

void PerformanceSession::soft_reset() {
    pipeline_.flush();
    seed_counter_.fetch_add(1);
}

bool PerformanceSession::is_playing() const {
    return playing_;
}

StreamControls PerformanceSession::build_controls() const {
    StreamControls c;
    ComposedPrompt composed = PromptCompositor::compose(liveset_.snapshot());
    c.caption         = composed.blended_caption.empty() ? "ambient electronic" : composed.blended_caption;
    c.denoise         = shared_.denoise_value();
    c.cover_strength  = shared_.cover_strength_value();
    c.guidance        = shared_.guidance_value();
    c.feedback        = shared_.feedback_value();
    c.feedback_depth  = shared_.feedback_depth_value();
    c.seed            = static_cast<std::int64_t>(seed_counter_.load());
    return c;
}

SessionView PerformanceSession::snapshot() const {
    SessionView view;
    view.prompts              = liveset_.snapshot();
    view.denoise              = shared_.denoise_value();
    view.cover_strength       = shared_.cover_strength_value();
    view.guidance             = shared_.guidance_value();
    view.feedback             = shared_.feedback_value();
    view.feedback_depth       = shared_.feedback_depth_value();
    view.pipeline             = pipeline_.stats();
    view.playing              = playing_;
    view.inference_ready      = inference_.ready();
    view.generating           = generating_;
    view.inference_error      = inference_.ready() ? "" : inference_.last_error();
    view.liveset_name         = liveset_.name();

    ComposedPrompt composed = PromptCompositor::compose(view.prompts);
    view.composed_caption    = composed.blended_caption;
    view.active_prompt_count = composed.active_count;
    return view;
}

void PerformanceSession::stream_loop() {
    using namespace std::chrono_literals;

    while (!stop_requested_) {
        generating_ = true;
        if (state_callback_) {
            state_callback_();
        }

        StreamTickResult tick = inference_.tick(build_controls(), pipeline_);

        generating_ = false;
        if (state_callback_) {
            state_callback_();
        }

        if (!tick.wav.empty() && audio_callback_) {
            audio_callback_(tick.wav);
        }

        if (tick.did_work) {
            std::this_thread::sleep_for(1ms);
        } else {
            std::this_thread::sleep_for(5ms);
        }
    }
    generating_ = false;
}

void PerformanceSession::simulate_encode(const PromptEntry & entry) {
    std::thread([this, id = entry.id] {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        liveset_.mark_encode_ready(id);
    }).detach();
}

}  // namespace aceimprovise
