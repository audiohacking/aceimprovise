# aceimprovise

Live streaming diffusion for ACE-Step — built for **performances**, not batch renders.

aceimprovise wraps [acestep.cpp](https://github.com/ServeurpersoCom/acestep.cpp) (git submodule) with a DEMON-inspired ring-buffer engine and a creator UI. Add, remove, and rewrite prompts **while audio keeps flowing**. There is **no cap** on prompt lanes — build as many layers as the set needs.

Design reference: [DEMON](https://daydreamlive.github.io/DEMON/) (theory only; this is a separate project).

## Quick start

```bash
git clone --recurse-submodules https://github.com/audiohacking/aceimprovise.git
cd aceimprovise

# Build C++ server (Metal on macOS)
chmod +x scripts/build-macos.sh
./scripts/build-macos.sh

# Web UI
cd web && npm install && npm run build && cd ..

# Run
./build/ace-improvise-server --static web/dist
# open http://127.0.0.1:8765
```

Dev UI with hot reload:

```bash
./build/ace-improvise-server --static web/dist &
cd web && npm run dev   # http://localhost:5173 proxies API to :8765
```

## Submodule sync

```bash
git submodule update --init --recursive
cd third_party/acestep.cpp && git fetch && git checkout master && git pull
cd ../.. && git add third_party/acestep.cpp && git commit -m "chore: bump acestep.cpp"
```

## Architecture

| Layer | Role |
|---|---|
| `third_party/acestep.cpp` | GGML inference (read-only submodule) |
| `src/aceimprovise/liveset/` | Unlimited prompt lanes, compositor |
| `src/aceimprovise/stream/` | Ring buffer + shared controls |
| `src/aceimprovise/session/` | Performance session (live API) |
| `src/aceimprovise/server/` | HTTP + WebSocket for UI |
| `web/` | Creator UI |

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Status

**v0 bootstrap** — liveset API, ring-buffer orchestration skeleton, web UI. Full acestep.cpp DiT/VAE streaming hooks land in the next milestones.

## License

MIT (aceimprovise). acestep.cpp and ACE-Step weights carry their upstream licenses.
