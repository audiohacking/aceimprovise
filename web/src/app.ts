import type { SessionState } from "./types";
import {
  RANDOM_VIBES,
  addPrompt,
  connectWebSocket,
  fetchState,
  play,
  removePrompt,
  reset,
  setShared,
  stop,
  updatePrompt,
} from "./api/client";

let state: SessionState | null = null;

const app = document.querySelector<HTMLDivElement>("#app")!;

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className?: string,
  text?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function render() {
  if (!state) return;

  app.innerHTML = "";

  const shell = el("div", "shell");

  const header = el("header", "header");
  const brand = el("div", "brand");
  brand.append(el("h1", "title", "aceimprovise"), el("span", "tagline", "live diffusion · unlimited prompts"));
  const status = el("div", "status");
  const statusParts: string[] = [];
  if (state.playing) {
    statusParts.push(state.generating ? "generating" : "streaming");
  } else {
    statusParts.push("idle");
  }
  if (state.inference_ready === false) {
    statusParts.push("no models");
  }
  status.append(
    el("span", state.playing ? "dot live" : "dot", ""),
    el("span", "status-text", statusParts.join(" · ")),
    el("span", "mono", `${state.prompt_count} prompts · ${state.active_prompt_count} active`),
  );
  if (state.inference_error) {
    status.append(el("span", "mono small error-hint", state.inference_error));
  }
  const transport = el("div", "transport");
  const playBtn = el("button", "btn primary", state.playing ? "■ Stop" : "▶ Play");
  playBtn.onclick = async () => {
    state = state?.playing ? await stop() : await play();
    render();
  };
  const resetBtn = el("button", "btn", "↺ Reset");
  resetBtn.onclick = async () => {
    state = await reset();
    render();
  };
  transport.append(playBtn, resetBtn);
  header.append(brand, status, transport);

  const grid = el("main", "grid");

  const sourcePanel = el("section", "panel source-panel");
  sourcePanel.append(
    el("h2", "panel-title", "Source"),
    el("p", "hint", "Drop a loop or pick a demo track. Cover mode keeps your groove while prompts morph the vibe."),
    el("div", "wave-placeholder", "waveform · coming soon"),
    el("p", "mono small", state.composed_caption || "Add prompts to build your blend"),
  );

  const promptPanel = el("section", "panel prompt-panel");
  promptPanel.append(el("h2", "panel-title", "Liveset"));

  const list = el("div", "prompt-list");
  state.prompts.forEach((prompt, index) => {
    const row = el("article", "prompt-row");
    const indexLabel = el("span", "prompt-index", `#${index + 1}`);
    const caption = el("input", "prompt-caption");
    caption.type = "text";
    caption.value = prompt.caption;
    caption.placeholder = "Describe this layer…";
    caption.onchange = () => {
      void updatePrompt(prompt.id, { caption: caption.value }).then((s) => {
        state = s;
        render();
      });
    };

    const slider = el("input", "prompt-weight") as HTMLInputElement;
    slider.type = "range";
    slider.min = "0";
    slider.max = "1";
    slider.step = "0.01";
    slider.value = String(prompt.weight);
    slider.oninput = () => {
      void updatePrompt(prompt.id, { weight: parseFloat(slider.value) });
    };
    slider.onchange = () => {
      void updatePrompt(prompt.id, { weight: parseFloat(slider.value) }).then((s) => {
        state = s;
        render();
      });
    };

    const muteBtn = el("button", "icon-btn", prompt.muted ? "🔇" : "🔊");
    muteBtn.title = prompt.muted ? "Unmute" : "Mute";
    muteBtn.onclick = () => {
      void updatePrompt(prompt.id, { muted: !prompt.muted }).then((s) => {
        state = s;
        render();
      });
    };

    const removeBtn = el("button", "icon-btn danger", "×");
    removeBtn.title = "Remove lane";
    removeBtn.onclick = () => {
      void removePrompt(prompt.id).then((s) => {
        state = s;
        render();
      });
    };

    if (prompt.encode_status === "pending") {
      row.classList.add("encoding");
    }

    row.append(indexLabel, caption, slider, muteBtn, removeBtn);
    list.append(row);
  });
  promptPanel.append(list);

  const addRow = el("div", "add-row");
  const input = el("input", "vibe-input") as HTMLInputElement;
  input.placeholder = "Describe your vibe…";
  input.onkeydown = (ev) => {
    if (ev.key === "Enter") addBtn.click();
  };
  const addBtn = el("button", "btn", "+ Add");
  addBtn.onclick = () => {
    const caption = input.value.trim();
    if (!caption) return;
    void addPrompt(caption).then((s) => {
      state = s;
      input.value = "";
      render();
    });
  };
  const randomBtn = el("button", "btn ghost", "🎲 Random");
  randomBtn.onclick = () => {
    const pick = RANDOM_VIBES[Math.floor(Math.random() * RANDOM_VIBES.length)];
    input.value = pick;
    void addPrompt(pick).then((s) => {
      state = s;
      input.value = "";
      render();
    });
  };
  addRow.append(input, addBtn, randomBtn);
  promptPanel.append(addRow);

  const feelPanel = el("section", "panel feel-panel");
  feelPanel.append(el("h2", "panel-title", "Feel"));

  const makeSlider = (
    label: string,
    value: number,
    min: number,
    max: number,
    key: "denoise" | "cover_strength" | "guidance" | "feedback",
  ) => {
    const wrap = el("label", "feel-control");
    wrap.append(el("span", "feel-label", label), el("span", "feel-value mono", value.toFixed(2)));
    const s = el("input") as HTMLInputElement;
    s.type = "range";
    s.min = String(min);
    s.max = String(max);
    s.step = "0.01";
    s.value = String(value);
    s.oninput = () => {
      void setShared({ [key]: parseFloat(s.value) });
    };
    s.onchange = () => {
      void setShared({ [key]: parseFloat(s.value) }).then((next) => {
        state = next;
        render();
      });
    };
    wrap.append(s);
    return wrap;
  };

  feelPanel.append(
    makeSlider("Denoise", state.shared.denoise, 0, 1, "denoise"),
    makeSlider("Cover", state.shared.cover_strength, 0, 1, "cover_strength"),
    makeSlider("Guidance", state.shared.guidance, 0, 3, "guidance"),
    makeSlider("Evolution", state.shared.feedback ?? 0.35, 0, 1, "feedback"),
    el("p", "mono small",
      `depth ${state.pipeline.depth} · active ${state.pipeline.active_slots ?? 0} · ticks ${state.pipeline.tick_count}`),
  );

  grid.append(sourcePanel, promptPanel, feelPanel);
  shell.append(header, grid);
  app.append(shell);
}

async function boot() {
  app.innerHTML = `<p class="loading">Connecting…</p>`;
  try {
    state = await fetchState();
    render();
    connectWebSocket((next) => {
      state = next;
      render();
    });
  } catch (err) {
    app.innerHTML = `<p class="error">Could not reach ace-improvise-server. Start it on port 8765.</p>`;
    console.error(err);
  }
}

boot();
