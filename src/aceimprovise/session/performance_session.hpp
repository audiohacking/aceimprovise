#pragma once

#include "aceimprovise/inference/stream_inference.hpp"
#include "aceimprovise/liveset/liveset.hpp"
#include "aceimprovise/liveset/prompt_entry.hpp"
#include "aceimprovise/stream/pipeline.hpp"
#include "aceimprovise/stream/shared_state.hpp"

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace aceimprovise {

using AudioCallback = std::function<void(const std::vector<std::uint8_t> & wav)>;
using StateCallback = std::function<void()>;

struct SessionConfig {
    int         pipeline_depth = 4;
    int         steps          = 8;
    std::string models_dir;
    std::string source_path;
    bool        keep_loaded = true;
};

struct SessionView {
    std::vector<PromptEntry> prompts;
    float                    denoise        = 0.7f;
    float                    cover_strength = 0.85f;
    float                    guidance       = 1.0f;
    float                    feedback       = 0.35f;
    int                      feedback_depth = 1;
    PipelineStats            pipeline;
    bool                     playing = false;
    bool                     inference_ready = false;
    bool                     generating = false;
    std::string              inference_error;
    std::string              liveset_name;
    std::string              composed_caption;
    int                      active_prompt_count = 0;
};

class PerformanceSession {
public:
    explicit PerformanceSession(SessionConfig config = {});
    ~PerformanceSession();

    void set_audio_callback(AudioCallback cb);
    void set_state_callback(StateCallback cb);

    Liveset &       liveset();
    const Liveset & liveset() const;

    PromptEntry prompt_add(const std::string & caption, float weight = 1.0f);
    bool        prompt_remove(const std::string & id);
    bool        prompt_modify(const std::string & id, const std::string & caption);
    bool        prompt_weight(const std::string & id, float weight);
    bool        prompt_mute(const std::string & id, bool muted);

    void set_shared_denoise(float v);
    void set_shared_cover(float v);
    void set_shared_guidance(float v);
    void set_shared_feedback(float v);
    void set_shared_feedback_depth(int v);

    void start();
    void stop();
    void soft_reset();

    bool is_playing() const;
    SessionView snapshot() const;

private:
    void stream_loop();
    void simulate_encode(const PromptEntry & entry);
    StreamControls build_controls() const;
    void bump_generation_if_playing();

    SessionConfig config_;

    Liveset         liveset_;
    SharedState     shared_;
    StreamPipeline  pipeline_;
    StreamInference inference_;

    AudioCallback audio_callback_;
    StateCallback state_callback_;

    mutable std::mutex session_mutex_;

    std::atomic<bool> playing_{false};
    std::atomic<bool> generating_{false};
    std::atomic<bool> stop_requested_{false};
    std::atomic<std::uint64_t> seed_counter_{1528};
    std::thread worker_;
};

}  // namespace aceimprovise
