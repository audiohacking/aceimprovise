#pragma once

#include "aceimprovise/session/performance_session.hpp"

#include <functional>
#include <vector>

namespace aceimprovise {

struct HttpServerConfig {
    std::string host        = "127.0.0.1";
    int         port        = 8765;
    std::string static_root = "web/dist";
};

class HttpServer {
public:
    using BroadcastFn = std::function<void(const std::string & json)>;

    explicit HttpServer(PerformanceSession & session, HttpServerConfig config = {});
    ~HttpServer();

    bool start();
    void stop();
    void broadcast_state();
    void broadcast_audio(const std::vector<std::uint8_t> & wav);

private:
    void register_routes();
    void notify_clients();

    PerformanceSession & session_;
    HttpServerConfig     config_;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace aceimprovise
