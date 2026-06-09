#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace aceimprovise {

struct SlotRequest {
    std::int64_t seed           = 0;
    float        denoise        = 1.0f;
    float        cover_strength = 1.0f;
    float        guidance       = 1.0f;
    float        feedback       = 0.0f;
    int          feedback_depth = 1;
    std::string  caption;
};

struct DenoiseSlot {
    std::vector<float> xt;
    int                step_idx         = 0;
    int                schedule_offset  = 0;
    int                num_steps        = 0;
    std::int64_t       seed             = 0;
    bool               switched_cover   = false;
    float              denoise          = 0.7f;
    float              cover_strength   = 0.85f;
    float              guidance         = 1.0f;
};

}  // namespace aceimprovise
