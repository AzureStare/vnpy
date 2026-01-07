let audioCtx: AudioContext | null = null;

function getCtx(): AudioContext {
  if (!audioCtx) {
    audioCtx = new AudioContext();
  }
  return audioCtx;
}

export async function ensureAudioReady(): Promise<void> {
  const ctx = getCtx();
  if (ctx.state === "suspended") {
    await ctx.resume();
  }
}

export async function playTone(freqHz: number, durationMs: number, gain: number = 0.08): Promise<void> {
  const ctx = getCtx();
  if (ctx.state === "suspended") {
    // Caller should ensure this is triggered by user gesture for best chance to succeed.
    await ctx.resume();
  }

  const osc = ctx.createOscillator();
  const g = ctx.createGain();
  osc.type = "sine";
  osc.frequency.value = freqHz;
  g.gain.value = Math.max(0, Math.min(1, gain));

  osc.connect(g);
  g.connect(ctx.destination);

  const now = ctx.currentTime;
  osc.start(now);
  osc.stop(now + durationMs / 1000);
}

export async function playOpenSound(): Promise<void> {
  await playTone(880, 110, 0.08);
}

export async function playCloseSound(): Promise<void> {
  await playTone(440, 140, 0.08);
}

