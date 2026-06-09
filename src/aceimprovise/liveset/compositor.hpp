#pragma once

#include "aceimprovise/liveset/prompt_entry.hpp"

#include <string>
#include <vector>

namespace aceimprovise {

struct ComposedPrompt {
    std::string blended_caption;
    float       total_weight = 0.0f;
    int         active_count = 0;
};

// Weighted blend of all active (non-muted, weight > 0) prompt lanes.
class PromptCompositor {
public:
    static ComposedPrompt compose(const std::vector<PromptEntry> & entries);
};

}  // namespace aceimprovise
