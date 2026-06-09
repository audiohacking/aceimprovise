#pragma once

#include <cstdint>
#include <memory>
#include <vector>

struct DiTGGML;

namespace aceimprovise {

// Runs one flow-matching Euler step for a batch of ring slots that share the
// same step index (same t_curr). Graph state is reused across ticks.
struct DitRingStepParams {
    int           batch_n       = 0;
    int           step_idx      = 0;
    int           num_steps     = 0;
    const float * schedule      = nullptr;
    float         guidance_scale = 1.0f;
    int           cover_steps   = -1;
    bool          use_batch_cfg = true;

    const float * xt            = nullptr;  // [batch_n * T * Oc]
    float *       xt_out        = nullptr;  // same layout; may alias xt
    float *       x0_out        = nullptr;  // set on final step only

    const float * context       = nullptr;  // [batch_n * T * ctx_ch]
    const float * context_switch = nullptr;
    const float * enc_hidden    = nullptr;  // [enc_S * H * batch_n]
    const float * enc_switch    = nullptr;
    int           enc_S         = 0;
    const int *   real_S        = nullptr;
    const int *   real_enc_S    = nullptr;
    const int *   real_enc_S_switch = nullptr;
    const int64_t * seeds       = nullptr;

    std::uint8_t * switched_cover = nullptr;  // per-batch cover context switch (0/1)
};

class DitRingStepper {
public:
    DitRingStepper();
    ~DitRingStepper();

    DitRingStepper(const DitRingStepper &)            = delete;
    DitRingStepper & operator=(const DitRingStepper &) = delete;

    bool bind(DiTGGML * model, int max_batch, int T, int enc_S, bool use_batch_cfg);
    void unbind();

    bool ready() const;
    bool step(const DitRingStepParams & params);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace aceimprovise
