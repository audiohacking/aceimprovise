#include "aceimprovise/stream/pipeline.hpp"

#include <algorithm>
#include <chrono>

namespace aceimprovise {

StreamPipeline::StreamPipeline(int depth, int steps)
    : depth_(std::max(1, depth)), steps_(std::max(1, steps)), slots_(static_cast<std::size_t>(depth_)) {
    stats_.depth = depth_;
    stats_.steps = steps_;
}

void StreamPipeline::set_depth(int depth) {
    depth_ = std::max(1, depth);
    stats_.depth = depth_;
    slots_.resize(static_cast<std::size_t>(depth_));
}

int StreamPipeline::depth() const {
    return depth_;
}

void StreamPipeline::set_steps(int steps) {
    steps_       = std::max(1, steps);
    stats_.steps = steps_;
}

int StreamPipeline::steps() const {
    return steps_;
}

void StreamPipeline::submit(SlotRequest request) {
    if (static_cast<int>(queue_.size()) >= depth_) {
        queue_.pop_front();
    }
    queue_.push_back(std::move(request));
    stats_.queue_depth = static_cast<int>(queue_.size());
}

void StreamPipeline::init_slot(std::size_t index, DenoiseSlot slot) {
    if (index < slots_.size()) {
        slots_[index] = std::move(slot);
    }
}

void StreamPipeline::drain_queue(const std::function<DenoiseSlot(const SlotRequest &)> & factory) {
    if (!factory) {
        return;
    }
    for (std::size_t i = 0; i < slots_.size(); ++i) {
        if (slots_[i].has_value() || queue_.empty()) {
            continue;
        }
        SlotRequest req = queue_.front();
        queue_.pop_front();
        slots_[i] = factory(req);
    }
    stats_.queue_depth = static_cast<int>(queue_.size());
}

void StreamPipeline::clear_slot(std::size_t index) {
    if (index < slots_.size()) {
        slots_[index].reset();
    }
}

void StreamPipeline::advance_slot(std::size_t index) {
    if (index < slots_.size() && slots_[index]) {
        slots_[index]->step_idx++;
    }
}

void StreamPipeline::record_finished() {
    stats_.finished_latents++;
}

std::vector<std::size_t> StreamPipeline::take_finished() {
    std::vector<std::size_t> finished;
    for (std::size_t i = 0; i < slots_.size(); ++i) {
        if (!slots_[i]) {
            continue;
        }
        DenoiseSlot & slot = *slots_[i];
        if (slot.num_steps > 0 && slot.step_idx >= slot.num_steps) {
            finished.push_back(i);
        }
    }
    return finished;
}

std::vector<StepGroup> StreamPipeline::step_groups() const {
    std::vector<StepGroup> groups;
    for (std::size_t i = 0; i < slots_.size(); ++i) {
        if (!slots_[i]) {
            continue;
        }
        const DenoiseSlot & slot = *slots_[i];
        if (slot.num_steps <= 0 || slot.step_idx >= slot.num_steps) {
            continue;
        }
        const int step = slot.step_idx;
        auto      it   = std::find_if(groups.begin(), groups.end(),
                                      [step](const StepGroup & g) { return g.step_idx == step; });
        if (it == groups.end()) {
            StepGroup g;
            g.step_idx = step;
            g.slot_indices.push_back(i);
            groups.push_back(std::move(g));
        } else {
            it->slot_indices.push_back(i);
        }
    }
    return groups;
}

DenoiseSlot * StreamPipeline::slot_at(std::size_t index) {
    if (index >= slots_.size() || !slots_[index]) {
        return nullptr;
    }
    return &(*slots_[index]);
}

const DenoiseSlot * StreamPipeline::slot_at(std::size_t index) const {
    if (index >= slots_.size() || !slots_[index]) {
        return nullptr;
    }
    return &(*slots_[index]);
}

bool StreamPipeline::tick_mark(double tick_ms) {
    stats_.tick_count++;
    stats_.last_tick_ms = tick_ms;

    int active = 0;
    for (const auto & s : slots_) {
        if (s) {
            active++;
        }
    }
    stats_.active_slots = active;
    stats_.streaming    = active > 0;
    return active > 0;
}

void StreamPipeline::flush() {
    for (auto & s : slots_) {
        s.reset();
    }
    queue_.clear();
    stats_.streaming     = false;
    stats_.active_slots  = 0;
    stats_.queue_depth   = 0;
}

PipelineStats StreamPipeline::stats() const {
    return stats_;
}

bool StreamPipeline::is_streaming() const {
    return stats_.streaming;
}

}  // namespace aceimprovise
