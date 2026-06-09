# aceimprovise architecture

## Purpose

Live performance tool: continuous cover/improv generation with **unbounded prompt lanes**.

Inspired by [DEMON](https://daydreamlive.github.io/DEMON/) streaming theory; implemented on unmodified [acestep.cpp](https://github.com/ServeurpersoCom/acestep.cpp).

## Control lanes

1. **Submission lane** — new `SlotRequest` entries enter the ring when a slot opens. Prompt add/remove/modify changes what the compositor blends into the next submission.
2. **Shared lane** — denoise, cover strength, guidance, evolution (**feedback**), and prompt blend weights apply on the next tick. Text conditioning is re-encoded when the composed caption changes.

Never flush the ring on prompt edits unless the performer explicitly resets.

## Liveset

- `Liveset` holds a `std::vector<PromptEntry>` — **no maximum size**.
- `PromptCompositor` blends active lanes by weight.
- Text encode runs through acestep `ops_encode_text` when the blended caption changes.

## Stream pipeline (ring buffer)

Depth is decoupled from denoising steps (`pipeline_depth` vs `steps`).

Each wall-clock **tick**:

1. Submit a fresh `SlotRequest` (capped queue drops stale submissions).
2. Fill empty ring slots with noise initialized from source + **evolution feedback**.
3. Group in-flight slots by DiT step index; run **one Euler step per group** via `DitRingStepper`.
4. When a slot completes all steps, VAE-decode its latent and push WAV to the WebSocket client.
5. Push the finished latent into **feedback history** so the next slot can blend toward the previous generation (DEMON-style continuity).

After warmup (~`depth` ticks), throughput approaches one finished latent per tick.

## Evolution stream

- **feedback** (0–1): blends the cover source latent with a prior finished latent when initializing new slots.
- **feedback_depth**: which history tap to use (1 = most recent).
- Together with **denoise**, this gives continuous morphing rather than unrelated full re-synths.

## Inference stack

- `StreamInference` — session prep via acestep ops (`ops_encode_src`, `ops_encode_text`, …).
- `DitRingStepper` — reuses acestep DiT graph; one batched forward per step group (no submodule patches).
- `PerformanceSession::stream_loop` — drives ticks while playing.

## UI

Web client: dynamic prompt list, shared feel controls (including **Evolution**), WebSocket session + binary WAV frames.

## Submodule policy

`third_party/acestep.cpp` is the only upstream inference source. Bump via git submodule pointer; do not edit files inside the submodule in this repo.
