export type EncodeStatus = "ready" | "pending" | "failed";

export interface PromptLane {
  id: string;
  caption: string;
  weight: number;
  muted: boolean;
  encode_status: EncodeStatus;
}

export interface SharedControls {
  denoise: number;
  cover_strength: number;
  guidance: number;
  feedback: number;
  feedback_depth: number;
}

export interface PipelineStats {
  depth: number;
  steps: number;
  streaming: boolean;
  tick_count: number;
  finished_latents: number;
  last_tick_ms: number;
  active_slots?: number;
  queue_depth?: number;
}

export interface SessionState {
  liveset_name: string;
  playing: boolean;
  composed_caption: string;
  active_prompt_count: number;
  prompt_count: number;
  prompts: PromptLane[];
  shared: SharedControls;
  pipeline: PipelineStats;
  inference_ready?: boolean;
  generating?: boolean;
  inference_error?: string;
}
