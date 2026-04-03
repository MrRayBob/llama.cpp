#pragma once

#include "common.h"
#include "http.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cctype>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <map>
#include <thread>

#include "server-http.h"

struct server_cors_proxy_res : server_http_res {
    std::function<void()> cleanup = nullptr;

    ~server_cors_proxy_res() override {
        if (cleanup) {
            cleanup();
        }
    }
};

template<typename T>
struct server_cors_proxy_pipe_t {
    std::mutex mutex;
    std::condition_variable cv;
    std::queue<T> queue;
    std::atomic<bool> writer_closed{false};
    std::atomic<bool> reader_closed{false};

    void close_write() {
        writer_closed.store(true, std::memory_order_relaxed);
        cv.notify_all();
    }

    void close_read() {
        reader_closed.store(true, std::memory_order_relaxed);
        cv.notify_all();
    }

    bool read(T & output, const std::function<bool()> & should_stop) {
        std::unique_lock<std::mutex> lk(mutex);
        constexpr auto poll_interval = std::chrono::milliseconds(500);
        while (true) {
            if (!queue.empty()) {
                output = std::move(queue.front());
                queue.pop();
                return true;
            }
            if (writer_closed.load()) {
                return false;
            }
            if (should_stop()) {
                close_read();
                return false;
            }
            cv.wait_for(lk, poll_interval);
        }
    }

    bool write(T && data) {
        std::lock_guard<std::mutex> lk(mutex);
        if (reader_closed.load()) {
            return false;
        }
        queue.push(std::move(data));
        cv.notify_one();
        return true;
    }
};

struct server_cors_proxy_msg_t {
    std::map<std::string, std::string> headers;
    int status = 0;
    std::string data;
    std::string content_type;
};

static std::string server_cors_proxy_to_lower_copy(const std::string & value) {
    std::string lowered(value.size(), '\0');
    std::transform(value.begin(), value.end(), lowered.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return lowered;
}

static bool server_cors_should_strip_response_header(const std::string & header_name) {
    if (header_name == "server" ||
        header_name == "transfer-encoding" ||
        header_name == "content-length" ||
        header_name == "keep-alive") {
        return true;
    }

    if (header_name.rfind("access-control-", 0) == 0) {
        return true;
    }

    return false;
}

static server_http_res_ptr proxy_request(const server_http_req & req, std::string method) {
    std::string target_url = req.get_param("url");
    common_http_url parsed_url = common_http_parse_url(target_url);

    if (parsed_url.host.empty()) {
        throw std::runtime_error("invalid target URL: missing host");
    }

    if (parsed_url.path.empty()) {
        parsed_url.path = "/";
    }

    if (!parsed_url.password.empty()) {
        throw std::runtime_error("authentication in target URL is not supported");
    }

    if (parsed_url.scheme != "http" && parsed_url.scheme != "https") {
        throw std::runtime_error("unsupported URL scheme in target URL: " + parsed_url.scheme);
    }

    SRV_INF("proxying %s request to %s://%s:%i%s\n", method.c_str(), parsed_url.scheme.c_str(), parsed_url.host.c_str(), parsed_url.port, parsed_url.path.c_str());

    std::map<std::string, std::string> headers;
    for (auto [key, value] : req.headers) {
        auto lowered = server_cors_proxy_to_lower_copy(key);
        if (string_starts_with(lowered, "x-proxy-header-")) {
            headers[key.substr(std::string("x-proxy-header-").size())] = value;
            continue;
        }

        if (lowered == "authorization" || lowered == "x-api-key") {
            continue;
        }

        headers[key] = value;
    }

    auto res = std::make_unique<server_cors_proxy_res>();
    auto pipe = std::make_shared<server_cors_proxy_pipe_t<server_cors_proxy_msg_t>>();

    auto cli = std::make_shared<httplib::ClientImpl>(parsed_url.host, parsed_url.port);
    if (parsed_url.scheme == "https") {
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
        cli.reset(new httplib::SSLClient(parsed_url.host, parsed_url.port));
#else
        throw std::runtime_error("HTTPS requested but CPPHTTPLIB_OPENSSL_SUPPORT is not defined");
#endif
    }

    cli->set_follow_location(true);
    cli->set_connection_timeout(5, 0);
    cli->set_write_timeout(600, 0);
    cli->set_read_timeout(600, 0);

    res->status = 500;
    res->cleanup = [pipe]() {
        pipe->close_read();
        pipe->close_write();
    };
    res->next = [pipe, should_stop = req.should_stop](std::string & out) -> bool {
        server_cors_proxy_msg_t msg;
        bool has_next = pipe->read(msg, should_stop);
        if (!msg.data.empty()) {
            out = std::move(msg.data);
        }
        return has_next;
    };

    httplib::ResponseHandler response_handler = [pipe](const httplib::Response & response) {
        server_cors_proxy_msg_t msg;
        msg.status = response.status;
        for (const auto & [key, value] : response.headers) {
            const auto lowered = server_cors_proxy_to_lower_copy(key);
            if (server_cors_should_strip_response_header(lowered)) {
                continue;
            }
            if (lowered == "content-type") {
                msg.content_type = value;
                continue;
            }
            msg.headers[key] = value;
        }
        return pipe->write(std::move(msg));
    };

    httplib::ContentReceiverWithProgress content_receiver = [pipe](const char * data, size_t data_length, size_t, size_t) {
        return pipe->write({{}, 0, std::string(data, data_length), ""});
    };

    httplib::Request dst_req;
    dst_req.method = method;
    dst_req.path = parsed_url.path;
    for (const auto & [key, value] : headers) {
        const auto lowered = server_cors_proxy_to_lower_copy(key);
        if (lowered == "accept-encoding" || lowered == "transfer-encoding") {
            continue;
        }
        if (lowered == "host") {
            bool is_default_port =
                (parsed_url.scheme == "https" && parsed_url.port == 443) ||
                (parsed_url.scheme == "http" && parsed_url.port == 80);
            dst_req.set_header(key, is_default_port ? parsed_url.host : parsed_url.host + ":" + std::to_string(parsed_url.port));
        } else {
            dst_req.set_header(key, value);
        }
    }
    dst_req.body = req.body;
    dst_req.response_handler = response_handler;
    dst_req.content_receiver = content_receiver;

    std::thread([cli, pipe, dst_req = std::move(dst_req)]() mutable {
        auto result = cli->send(std::move(dst_req));
        if (result.error() != httplib::Error::Success) {
            auto err_str = httplib::to_string(result.error());
            SRV_ERR("http client error: %s\n", err_str.c_str());
            pipe->write({{}, 500, "", ""});
            pipe->write({{}, 0, "proxy error: " + err_str, ""});
        }
        pipe->close_write();
    }).detach();

    server_cors_proxy_msg_t header;
    if (pipe->read(header, req.should_stop)) {
        res->status = header.status;
        res->headers = std::move(header.headers);
        if (!header.content_type.empty()) {
            res->content_type = std::move(header.content_type);
        }
    }

    return res;
}

static server_http_context::handler_t proxy_handler_post = [](const server_http_req & req) -> server_http_res_ptr {
    return proxy_request(req, "POST");
};

static server_http_context::handler_t proxy_handler_get = [](const server_http_req & req) -> server_http_res_ptr {
    return proxy_request(req, "GET");
};
