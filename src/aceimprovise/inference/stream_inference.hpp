#pragma once

#include "aceimprovise/inference/dit_ring_stepper.hpp"
#include "aceimprovise/stream/pipeline.hpp"
#include "aceimprovise/stream/slot.hpp"

#include "request.h"

#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <vector>

struct AceSynth;
struct ModelStore;
struct ModelHandle;
struct SynthState;

namespace aceimprovise {

struct StreamConfig {
    std::string models_dir;
    std::string source_path;
    bool        keep_loaded = true;
    int         steps       = 8;
    int         max_depth   = 4;
};

struct StreamControls {
    std::string caption;
    float       denoise        = 0.7f;
    float       cover_strength = 0.85f;
    float       guidance       = 1.0f;
    float       feedback       = 0.35f;
    int         feedback_depth = 1;
    std::int64_t seed          = 1528;
};

struct StreamTickResult {
    bool                         did_work = false;
    std::optional<std::uint64_t> finished_slot;
    std::vector<std::uint8_t>    wav;
    double                       tick_ms = 0.0;
};

class StreamInference {
public:
    StreamInference();
    ~StreamInference();

    StreamInference(const StreamInference &)            = delete;
    StreamInference & operator=(const StreamInference &) = delete;

    bool init(const StreamConfig & config);
    bool ready() const;
    std::string last_error() const;

    StreamTickResult tick(const StreamControls & controls, StreamPipeline & pipeline);

private:
    bool ensure_synth();
    bool ensure_session_prepared();
    bool refresh_conditioning(const StreamControls & controls);
    bool ensure_dit_bound();
    SlotRequest make_submission(const StreamControls & controls) const;
    void push_latent_history(const std::vector<float> & latent);
    const std::vector<float> * feedback_latent(int depth) const;
    std::vector<float> build_evolved_source(const StreamControls & controls) const;
    bool init_slot_state(DenoiseSlot & slot, const SlotRequest & req, const StreamControls & controls);
    bool run_dit_tick(class StreamPipeline & pipeline, const StreamControls & controls);
    std::vector<std::uint8_t> decode_latent(const std::vector<float> & latent);

    StreamConfig config_;
    std::string  error_;

    ModelStore * store_ = nullptr;
    AceSynth *   synth_ = nullptr;
    bool         ready_ = false;

    std::unique_ptr<SynthState> session_;
    std::unique_ptr<ModelHandle> dit_holder_;
    DitRingStepper               stepper_;

    float * source_interleaved_ = nullptr;
    int     source_len_         = 0;

    std::string last_caption_;
    std::deque<std::vector<float>> latent_history_;

    std::vector<float> batch_xt_;
    std::vector<float> batch_x0_;
    std::vector<std::uint8_t> batch_switched_;
};

}  // namespace aceimprovise
