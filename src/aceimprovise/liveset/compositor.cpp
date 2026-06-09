#include "aceimprovise/liveset/compositor.hpp"

#include <cmath>
#include <sstream>

namespace aceimprovise {

ComposedPrompt PromptCompositor::compose(const std::vector<PromptEntry> & entries) {
    ComposedPrompt out;
    std::ostringstream blend;

    float weight_sum = 0.0f;
    bool  first      = true;

    for (const PromptEntry & entry : entries) {
        if (entry.muted || entry.weight <= 0.0f || entry.caption.empty()) {
            continue;
        }
        if (entry.encode_status == EncodeStatus::Failed) {
            continue;
        }

        if (!first) {
            blend << " | ";
        }
        first = false;

        blend << entry.caption;
        weight_sum += entry.weight;
        out.active_count++;
    }

    out.blended_caption = blend.str();
    out.total_weight    = weight_sum;
    return out;
}

}  // namespace aceimprovise
