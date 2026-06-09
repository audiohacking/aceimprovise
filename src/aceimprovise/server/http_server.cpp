#include "aceimprovise/server/http_server.hpp"

#include "aceimprovise/server/protocol.hpp"

#ifdef __GNUC__
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wshadow"
#endif
#include "httplib.h"
#ifdef __GNUC__
#    pragma GCC diagnostic pop
#endif

#include "yyjson.h"

#include <atomic>
#include <filesystem>
#include <mutex>
#include <set>
#include <thread>

namespace fs = std::filesystem;

namespace aceimprovise {

struct HttpServer::Impl {
    httplib::Server                        server;
    std::mutex                             ws_mutex;
    std::set<httplib::ws::WebSocket *>     clients;
    std::atomic<bool>                      running{false};
    std::thread                            server_thread;
};

HttpServer::HttpServer(PerformanceSession & session, HttpServerConfig config)
    : session_(session), config_(std::move(config)), impl_(std::make_unique<Impl>()) {}

HttpServer::~HttpServer() = default;

namespace {

std::string read_body_field(const std::string & body, const char * key) {
    yyjson_doc * doc = yyjson_read(body.c_str(), body.size(), 0);
    if (!doc) {
        return {};
    }
    yyjson_val * root = yyjson_doc_get_root(doc);
    yyjson_val * val  = yyjson_obj_get(root, key);
    std::string out;
    if (val && yyjson_is_str(val)) {
        out = yyjson_get_str(val);
    }
    yyjson_doc_free(doc);
    return out;
}

float read_body_float(const std::string & body, const char * key, float fallback) {
    yyjson_doc * doc = yyjson_read(body.c_str(), body.size(), 0);
    if (!doc) {
        return fallback;
    }
    yyjson_val * root = yyjson_doc_get_root(doc);
    yyjson_val * val  = yyjson_obj_get(root, key);
    float out         = fallback;
    if (val && yyjson_is_uint(val)) {
        out = static_cast<float>(yyjson_get_uint(val));
    } else if (val && yyjson_is_int(val)) {
        out = static_cast<float>(yyjson_get_int(val));
    } else if (val && yyjson_is_real(val)) {
        out = static_cast<float>(yyjson_get_real(val));
    }
    yyjson_doc_free(doc);
    return out;
}

bool read_body_bool(const std::string & body, const char * key, bool fallback) {
    yyjson_doc * doc = yyjson_read(body.c_str(), body.size(), 0);
    if (!doc) {
        return fallback;
    }
    yyjson_val * root = yyjson_doc_get_root(doc);
    yyjson_val * val  = yyjson_obj_get(root, key);
    bool out          = fallback;
    if (val && yyjson_is_bool(val)) {
        out = yyjson_get_bool(val);
    }
    yyjson_doc_free(doc);
    return out;
}

}  // namespace

bool HttpServer::start() {
    register_routes();

    impl_->server.set_pre_routing_handler([](const httplib::Request &, httplib::Response & res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, PATCH, DELETE, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type");
        return httplib::Server::HandlerResponse::Unhandled;
    });

    impl_->server.Options(R"(.*)", [](const httplib::Request &, httplib::Response & res) {
        res.status = 204;
    });

    impl_->running = true;
    impl_->server_thread =
        std::thread([this] { impl_->server.listen(config_.host.c_str(), config_.port); });
    return true;
}

void HttpServer::stop() {
    if (!impl_->running.exchange(false)) {
        return;
    }
    impl_->server.stop();
    if (impl_->server_thread.joinable()) {
        impl_->server_thread.join();
    }
}

void HttpServer::broadcast_state() {
    notify_clients();
}

void HttpServer::broadcast_audio(const std::vector<std::uint8_t> & wav) {
    if (wav.empty()) {
        return;
    }

    yyjson_mut_doc * doc = yyjson_mut_doc_new(nullptr);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_obj_add_str(doc, root, "type", "audio");
    yyjson_mut_obj_add_str(doc, root, "mime", "audio/wav");
    yyjson_mut_obj_add_int(doc, root, "size", static_cast<int>(wav.size()));

    const char * json = yyjson_mut_write(doc, YYJSON_WRITE_NOFLAG, nullptr);
    std::string header = json ? json : "{}";
    if (json) {
        free((void *) json);
    }
    yyjson_mut_doc_free(doc);

    std::lock_guard<std::mutex> lock(impl_->ws_mutex);
    for (httplib::ws::WebSocket * ws : impl_->clients) {
        ws->send(header);
        ws->send(reinterpret_cast<const char *>(wav.data()), wav.size());
    }
}

void HttpServer::notify_clients() {
    const std::string payload = session_to_json(session_.snapshot());
    std::lock_guard<std::mutex> lock(impl_->ws_mutex);
    for (httplib::ws::WebSocket * ws : impl_->clients) {
        ws->send(payload);
    }
}

void HttpServer::register_routes() {
    const fs::path static_root = config_.static_root;

    impl_->server.Get("/api/state", [this](const httplib::Request &, httplib::Response & res) {
        res.set_content(session_to_json(session_.snapshot()), "application/json");
    });

    impl_->server.Post("/api/prompts", [this](const httplib::Request & req, httplib::Response & res) {
        const std::string caption = read_body_field(req.body, "caption");
        if (caption.empty()) {
            res.status = 400;
            res.set_content(error_json("caption required"), "application/json");
            return;
        }
        const float weight = read_body_float(req.body, "weight", 1.0f);
        session_.prompt_add(caption, weight);
        notify_clients();
        res.set_content(session_to_json(session_.snapshot()), "application/json");
    });

    impl_->server.Patch(R"(/api/prompts/([A-Za-z0-9]+))", [this](const httplib::Request & req,
                                                                  httplib::Response &      res) {
        const std::string id = req.matches[1];
        if (req.body.find("caption") != std::string::npos) {
            const std::string caption = read_body_field(req.body, "caption");
            if (!session_.prompt_modify(id, caption)) {
                res.status = 404;
                res.set_content(error_json("prompt not found"), "application/json");
                return;
            }
        }
        if (req.body.find("weight") != std::string::npos) {
            session_.prompt_weight(id, read_body_float(req.body, "weight", 0.0f));
        }
        if (req.body.find("muted") != std::string::npos) {
            session_.prompt_mute(id, read_body_bool(req.body, "muted", false));
        }
        notify_clients();
        res.set_content(session_to_json(session_.snapshot()), "application/json");
    });

    impl_->server.Delete(R"(/api/prompts/([A-Za-z0-9]+))", [this](const httplib::Request & req,
                                                                    httplib::Response &      res) {
        const std::string id = req.matches[1];
        if (!session_.prompt_remove(id)) {
            res.status = 404;
            res.set_content(error_json("prompt not found"), "application/json");
            return;
        }
        notify_clients();
        res.set_content(session_to_json(session_.snapshot()), "application/json");
    });

    impl_->server.Post("/api/shared", [this](const httplib::Request & req, httplib::Response & res) {
        if (req.body.find("denoise") != std::string::npos) {
            session_.set_shared_denoise(read_body_float(req.body, "denoise", 0.7f));
        }
        if (req.body.find("cover_strength") != std::string::npos) {
            session_.set_shared_cover(read_body_float(req.body, "cover_strength", 0.85f));
        }
        if (req.body.find("guidance") != std::string::npos) {
            session_.set_shared_guidance(read_body_float(req.body, "guidance", 1.0f));
        }
        if (req.body.find("feedback") != std::string::npos) {
            session_.set_shared_feedback(read_body_float(req.body, "feedback", 0.35f));
        }
        if (req.body.find("feedback_depth") != std::string::npos) {
            session_.set_shared_feedback_depth(static_cast<int>(read_body_float(req.body, "feedback_depth", 1.0f)));
        }
        notify_clients();
        res.set_content(session_to_json(session_.snapshot()), "application/json");
    });

    impl_->server.Post("/api/play", [this](const httplib::Request &, httplib::Response & res) {
        session_.start();
        notify_clients();
        res.set_content(session_to_json(session_.snapshot()), "application/json");
    });

    impl_->server.Post("/api/stop", [this](const httplib::Request &, httplib::Response & res) {
        session_.stop();
        notify_clients();
        res.set_content(session_to_json(session_.snapshot()), "application/json");
    });

    impl_->server.Post("/api/reset", [this](const httplib::Request &, httplib::Response & res) {
        session_.soft_reset();
        notify_clients();
        res.set_content(session_to_json(session_.snapshot()), "application/json");
    });

    impl_->server.WebSocket("/ws", [this](const httplib::Request &, httplib::ws::WebSocket & ws) {
        {
            std::lock_guard<std::mutex> lock(impl_->ws_mutex);
            impl_->clients.insert(&ws);
        }
        ws.send(session_to_json(session_.snapshot()));
        while (ws.is_open()) {
            std::string msg;
            if (ws.read(msg) == httplib::ws::Fail) {
                break;
            }
        }
        std::lock_guard<std::mutex> lock(impl_->ws_mutex);
        impl_->clients.erase(&ws);
    });

    if (fs::is_directory(static_root)) {
        impl_->server.set_mount_point("/", static_root.string());
    }
}

}  // namespace aceimprovise
