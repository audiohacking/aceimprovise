let ctx: AudioContext | null = null;
let activeSource: AudioBufferSourceNode | null = null;
let queueTail = Promise.resolve();

function getContext(): AudioContext {
  if (!ctx) {
    ctx = new AudioContext();
  }
  return ctx;
}

async function decodeAndPlay(bytes: ArrayBuffer): Promise<void> {
  const ac = getContext();
  if (ac.state === "suspended") {
    await ac.resume();
  }

  const buffer = await ac.decodeAudioData(bytes.slice(0));

  if (activeSource) {
    try {
      activeSource.stop();
    } catch {
      /* already stopped */
    }
    activeSource.disconnect();
    activeSource = null;
  }

  const source = ac.createBufferSource();
  source.buffer = buffer;
  source.connect(ac.destination);
  source.start(0);
  activeSource = source;

  await new Promise<void>((resolve) => {
    source.onended = () => resolve();
  });
}

/** Queue WAV playback so chunks do not overlap decode. */
export function playWavBytes(bytes: ArrayBuffer): void {
  queueTail = queueTail
    .then(() => decodeAndPlay(bytes))
    .catch((err) => console.error("[aceimprovise] audio playback failed", err));
}

export function stopPlayback(): void {
  if (activeSource) {
    try {
      activeSource.stop();
    } catch {
      /* ignore */
    }
    activeSource.disconnect();
    activeSource = null;
  }
}
