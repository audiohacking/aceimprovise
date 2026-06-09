#include "aceimprovise/liveset/liveset.hpp"

#include "aceimprovise/util/id.hpp"

#include <algorithm>

namespace aceimprovise {

namespace {

PromptEntry * find_entry(std::vector<PromptEntry> & entries, const std::string & id) {
    auto it = std::find_if(entries.begin(), entries.end(), [&](const PromptEntry & e) {
        return e.id == id;
    });
    return it == entries.end() ? nullptr : &(*it);
}

const PromptEntry * find_entry(const std::vector<PromptEntry> & entries, const std::string & id) {
    auto it = std::find_if(entries.begin(), entries.end(), [&](const PromptEntry & e) {
        return e.id == id;
    });
    return it == entries.end() ? nullptr : &(*it);
}

}  // namespace

std::vector<PromptEntry> Liveset::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return entries_;
}

PromptEntry Liveset::add(const std::string & caption, float initial_weight) {
    PromptEntry entry;
    entry.id      = next_prompt_id();
    entry.caption = caption;
    entry.weight  = initial_weight;
    entry.muted   = false;
    entry.encode_status = EncodeStatus::Pending;

    std::lock_guard<std::mutex> lock(mutex_);
    entries_.push_back(entry);
    return entry;
}

bool Liveset::remove(const std::string & id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = std::remove_if(entries_.begin(), entries_.end(), [&](const PromptEntry & e) {
        return e.id == id;
    });
    if (it == entries_.end()) {
        return false;
    }
    entries_.erase(it, entries_.end());
    return true;
}

bool Liveset::modify(const std::string & id, const std::string & caption) {
    std::lock_guard<std::mutex> lock(mutex_);
    PromptEntry * entry = find_entry(entries_, id);
    if (!entry) {
        return false;
    }
    entry->caption = caption;
    entry->encode_status = EncodeStatus::Pending;
    return true;
}

bool Liveset::set_weight(const std::string & id, float weight) {
    std::lock_guard<std::mutex> lock(mutex_);
    PromptEntry * entry = find_entry(entries_, id);
    if (!entry) {
        return false;
    }
    entry->weight = weight;
    return true;
}

bool Liveset::set_muted(const std::string & id, bool muted) {
    std::lock_guard<std::mutex> lock(mutex_);
    PromptEntry * entry = find_entry(entries_, id);
    if (!entry) {
        return false;
    }
    entry->muted = muted;
    return true;
}

bool Liveset::mark_encode_ready(const std::string & id) {
    std::lock_guard<std::mutex> lock(mutex_);
    PromptEntry * entry = find_entry(entries_, id);
    if (!entry) {
        return false;
    }
    entry->encode_status = EncodeStatus::Ready;
    return true;
}

std::optional<PromptEntry> Liveset::get(const std::string & id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const PromptEntry * entry = find_entry(entries_, id);
    if (!entry) {
        return std::nullopt;
    }
    return *entry;
}

std::size_t Liveset::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return entries_.size();
}

void Liveset::set_name(const std::string & name) {
    std::lock_guard<std::mutex> lock(mutex_);
    name_ = name;
}

std::string Liveset::name() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return name_;
}

}  // namespace aceimprovise
