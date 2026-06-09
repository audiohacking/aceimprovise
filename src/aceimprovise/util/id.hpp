#pragma once

#include <atomic>
#include <cstdint>
#include <string>

namespace aceimprovise {

inline std::string next_prompt_id() {
    static std::atomic<std::uint64_t> counter{1};
    return "p" + std::to_string(counter.fetch_add(1));
}

}  // namespace aceimprovise
