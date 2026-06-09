import type { SessionState } from "../types";
import { playWavBytes, stopPlayback } from "../audio/player";

const API_BASE =
  typeof window !== "undefined" && window.location.port === "5173"
    ? ""
    : "";

export async function fetchState(): Promise<SessionState> {
  const res = await fetch(`${API_BASE}/api/state`);
  if (!res.ok) throw new Error("Failed to fetch state");
  return res.json();
}

export async function addPrompt(caption: string, weight = 1): Promise<SessionState> {
  const res = await fetch(`${API_BASE}/api/prompts`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ caption, weight }),
  });
  if (!res.ok) throw new Error("Failed to add prompt");
  return res.json();
}

export async function updatePrompt(
  id: string,
  patch: Partial<{ caption: string; weight: number; muted: boolean }>,
): Promise<SessionState> {
  const res = await fetch(`${API_BASE}/api/prompts/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });
  if (!res.ok) throw new Error("Failed to update prompt");
  return res.json();
}

export async function removePrompt(id: string): Promise<SessionState> {
  const res = await fetch(`${API_BASE}/api/prompts/${id}`, { method: "DELETE" });
  if (!res.ok) throw new Error("Failed to remove prompt");
  return res.json();
}

export async function setShared(
  patch: Partial<{
    denoise: number;
    cover_strength: number;
    guidance: number;
    feedback: number;
    feedback_depth: number;
  }>,
): Promise<SessionState> {
  const res = await fetch(`${API_BASE}/api/shared`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });
  if (!res.ok) throw new Error("Failed to update shared controls");
  return res.json();
}

export async function play(): Promise<SessionState> {
  const res = await fetch(`${API_BASE}/api/play`, { method: "POST" });
  return res.json();
}

export async function stop(): Promise<SessionState> {
  stopPlayback();
  const res = await fetch(`${API_BASE}/api/stop`, { method: "POST" });
  return res.json();
}

export async function reset(): Promise<SessionState> {
  const res = await fetch(`${API_BASE}/api/reset`, { method: "POST" });
  return res.json();
}

export function connectWebSocket(onState: (state: SessionState) => void): () => void {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  const host =
    window.location.port === "5173"
      ? "127.0.0.1:8765"
      : window.location.host;
  const ws = new WebSocket(`${proto}//${host}/ws`);
  ws.binaryType = "arraybuffer";

  let pendingAudioBytes: number | null = null;

  ws.onmessage = (ev) => {
    if (typeof ev.data === "string") {
      try {
        const msg = JSON.parse(ev.data) as SessionState & { type?: string; size?: number };
        if (msg.type === "audio" && typeof msg.size === "number") {
          pendingAudioBytes = msg.size;
          return;
        }
        onState(msg);
      } catch {
        /* ignore malformed frames */
      }
      return;
    }

    if (ev.data instanceof ArrayBuffer && pendingAudioBytes !== null) {
      if (ev.data.byteLength === pendingAudioBytes) {
        playWavBytes(ev.data);
      }
      pendingAudioBytes = null;
    }
  };

  return () => ws.close();
}

export const RANDOM_VIBES = [
  "warm tape hiss and dusty chords",
  "broken amen break, sub pressure",
  "mono bass drone, slow filter sweep",
  "sparkling arpeggios, wide stereo pad",
  "industrial techno, distorted kick",
  "jazz piano cluster, brushed snare",
  "hypnotic dub chord stab",
  "glitchy granular texture, no drums",
  "euphoric trance supersaw lift",
  "dark ambient reverb tail",
];
