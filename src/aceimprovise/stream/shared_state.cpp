#include "aceimprovise/stream/shared_state.hpp"

namespace aceimprovise {

void SharedState::set_denoise(float v) {
    std::lock_guard<std::mutex> lock(mutex_);
    denoise_ = v;
}

void SharedState::set_cover_strength(float v) {
    std::lock_guard<std::mutex> lock(mutex_);
    cover_strength_ = v;
}

void SharedState::set_guidance(float v) {
    std::lock_guard<std::mutex> lock(mutex_);
    guidance_ = v;
}

void SharedState::set_feedback(float v) {
    std::lock_guard<std::mutex> lock(mutex_);
    feedback_ = v;
}

void SharedState::set_feedback_depth(int v) {
    std::lock_guard<std::mutex> lock(mutex_);
    feedback_depth_ = std::max(1, v);
}

float SharedState::denoise_value() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return denoise_;
}

float SharedState::cover_strength_value() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cover_strength_;
}

float SharedState::guidance_value() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return guidance_;
}

float SharedState::feedback_value() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return feedback_;
}

int SharedState::feedback_depth_value() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return feedback_depth_;
}

}  // namespace aceimprovise
