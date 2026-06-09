#pragma once

#include "aceimprovise/liveset/prompt_entry.hpp"

#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace aceimprovise {

// Dynamic liveset: unbounded prompt lanes for live performance.
class Liveset {
public:
    std::vector<PromptEntry> snapshot() const;

    PromptEntry add(const std::string & caption, float initial_weight = 1.0f);
    bool remove(const std::string & id);
    bool modify(const std::string & id, const std::string & caption);
    bool set_weight(const std::string & id, float weight);
    bool set_muted(const std::string & id, bool muted);
    bool mark_encode_ready(const std::string & id);

    std::optional<PromptEntry> get(const std::string & id) const;
    std::size_t size() const;

    void set_name(const std::string & name);
    std::string name() const;

private:
    mutable std::mutex       mutex_;
    std::string              name_ = "untitled";
    std::vector<PromptEntry> entries_;
};

}  // namespace aceimprovise
