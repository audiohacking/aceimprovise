#pragma once

#include <cstdint>
#include <string>

namespace aceimprovise {

enum class EncodeStatus : std::uint8_t {
    Ready = 0,
    Pending,
    Failed,
};

// One prompt lane in a liveset. There is no fixed upper bound on how many
// entries a session may hold — performers add lanes as inspiration strikes.
struct PromptEntry {
    std::string id;
    std::string caption;
    float       weight = 0.0f;
    bool        muted  = false;
    EncodeStatus encode_status = EncodeStatus::Ready;
};

}  // namespace aceimprovise
