#pragma once

#include <mutex>

namespace aceimprovise {

// Shared-lane controls: every in-flight ring slot reads these on the next tick.
struct SharedState {
    float denoise        = 0.7f;
    float cover_strength = 0.85f;
    float guidance       = 1.0f;
    float feedback       = 0.35f;
    int   feedback_depth = 1;

    void set_denoise(float v);
    void set_cover_strength(float v);
    void set_guidance(float v);
    void set_feedback(float v);
    void set_feedback_depth(int v);

    float denoise_value() const;
    float cover_strength_value() const;
    float guidance_value() const;
    float feedback_value() const;
    int   feedback_depth_value() const;

private:
    mutable std::mutex mutex_;
    float denoise_        = 0.7f;
    float cover_strength_ = 0.85f;
    float guidance_       = 1.0f;
    float feedback_       = 0.35f;
    int   feedback_depth_ = 1;
};

}  // namespace aceimprovise
