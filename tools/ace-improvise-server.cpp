#include "aceimprovise/server/http_server.hpp"
#include "aceimprovise/session/performance_session.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <thread>

namespace fs = std::filesystem;

namespace {

std::atomic<bool> g_running{true};

void on_signal(int) {
    g_running = false;
}

void usage(const char * argv0) {
    std::fprintf(stderr,
                 "Usage: %s [--host ADDR] [--port N] [--static DIR] [--models DIR] [--source WAV]\n",
                 argv0);
}

}  // namespace

int main(int argc, char ** argv) {
    aceimprovise::SessionConfig config;
    aceimprovise::HttpServerConfig server_cfg;

    for (int i = 1; i < argc; ++i) {
        const char * arg = argv[i];
        if (std::strcmp(arg, "--host") == 0 && i + 1 < argc) {
            server_cfg.host = argv[++i];
        } else if (std::strcmp(arg, "--port") == 0 && i + 1 < argc) {
            server_cfg.port = std::atoi(argv[++i]);
        } else if (std::strcmp(arg, "--static") == 0 && i + 1 < argc) {
            server_cfg.static_root = argv[++i];
        } else if (std::strcmp(arg, "--models") == 0 && i + 1 < argc) {
            config.models_dir = argv[++i];
        } else if (std::strcmp(arg, "--source") == 0 && i + 1 < argc) {
            config.source_path = argv[++i];
        } else if (std::strcmp(arg, "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    if (config.models_dir.empty()) {
        const fs::path default_models = "third_party/acestep.cpp/models";
        if (fs::is_directory(default_models)) {
            config.models_dir = default_models.string();
        }
    }

    std::signal(SIGINT, on_signal);
#ifndef _WIN32
    std::signal(SIGTERM, on_signal);
#endif

    aceimprovise::PerformanceSession session(config);
    aceimprovise::HttpServer         server(session, server_cfg);

    session.set_audio_callback([&server](const std::vector<std::uint8_t> & wav) {
        server.broadcast_audio(wav);
    });
    session.set_state_callback([&server] { server.broadcast_state(); });

    if (!server.start()) {
        std::fprintf(stderr, "Failed to start server\n");
        return 1;
    }

    const auto view = session.snapshot();
    std::printf("aceimprovise listening on http://%s:%d\n", server_cfg.host.c_str(), server_cfg.port);
    if (view.inference_ready) {
        std::printf("  inference: ready (%s)\n", config.models_dir.c_str());
        if (!config.source_path.empty()) {
            std::printf("  source: %s (cover mode)\n", config.source_path.c_str());
        } else {
            std::printf("  source: none (text2music mode)\n");
        }
    } else {
        std::printf("  inference: stub mode — %s\n", view.inference_error.c_str());
        std::printf("  hint: run scripts/fetch-models.sh and pass --models DIR\n");
    }

    while (g_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    session.stop();
    server.stop();
    return 0;
}
