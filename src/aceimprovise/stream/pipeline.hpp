#pragma once

#include "aceimprovise/stream/slot.hpp"

#include <cstdint>
#include <deque>
#include <functional>
#include <optional>
#include <vector>

namespace aceimprovise {

struct PipelineStats {
    std::uint64_t tick_count       = 0;
    std::uint64_t finished_latents = 0;
    int           depth            = 4;
    int           steps            = 8;
    bool          streaming        = false;
    double        last_tick_ms     = 0.0;
    int           active_slots     = 0;
    int           queue_depth      = 0;
};

struct StepGroup {
    int                      step_idx = 0;
    std::vector<std::size_t> slot_indices;
};

class StreamPipeline {
public:
    explicit StreamPipeline(int depth = 4, int steps = 8);

    void set_depth(int depth);
    int  depth() const;
    void set_steps(int steps);
    int  steps() const;

    void submit(SlotRequest request);
    void init_slot(std::size_t index, DenoiseSlot slot);

    // Fill empty ring positions from the capped submission queue.
    void drain_queue(const std::function<DenoiseSlot(const SlotRequest &)> & factory);

    void clear_slot(std::size_t index);
    void advance_slot(std::size_t index);
    void record_finished();

    // Collect finished slots before advancing (step_idx >= num_steps - 1).
    std::vector<std::size_t> take_finished();

    // Group active slots by DiT step index for batched forwards.
    std::vector<StepGroup> step_groups() const;

    DenoiseSlot *       slot_at(std::size_t index);
    const DenoiseSlot * slot_at(std::size_t index) const;

    bool tick_mark(double tick_ms);

    void flush();

    PipelineStats stats() const;
    bool          is_streaming() const;

private:
    int depth_;
    int steps_;

    std::vector<std::optional<DenoiseSlot>> slots_;
    std::deque<SlotRequest>                 queue_;

    PipelineStats stats_;
};

}  // namespace aceimprovise
