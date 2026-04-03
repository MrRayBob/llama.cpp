#include "arg.h"
#include "common.h"
#include "common/http.h"
#include "log.h"
#include "server-common.h"
#include "server-cors-proxy.h"
#include "server-http.h"

#include <cpp-httplib/httplib.h>

#include <algorithm>
#include <atomic>
#include <clocale>
#include <cctype>
#include <csignal>
#include <cstdint>
#include <functional>
#include <list>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <regex>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

struct proxy_options {
    std::string backend_base_url = "http://127.0.0.1:8080";
    std::string backend_api_key;
    std::string model_alias;
    int hard_prompt_cap = 32000;
    int compaction_trigger = 24000;
    std::unordered_map<std::string, int> model_hard_prompt_caps;
    std::unordered_map<std::string, int> model_compaction_triggers;
    int recent_raw_tail = 8;
    int first_summary_target = 800;
    int second_summary_target = 400;
    int summary_cache_capacity = 128;
};

struct summary_cache_entry {
    std::string key;
    std::string value;
};

class summary_cache {
public:
    explicit summary_cache(size_t capacity) : capacity_(std::max<size_t>(1, capacity)) {}

    bool get(const std::string & key, std::string & value) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = index_.find(key);
        if (it == index_.end()) {
            return false;
        }

        entries_.splice(entries_.begin(), entries_, it->second);
        value = it->second->value;
        return true;
    }

    void put(std::string key, std::string value) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = index_.find(key);
        if (it != index_.end()) {
            it->second->value = std::move(value);
            entries_.splice(entries_.begin(), entries_, it->second);
            return;
        }

        entries_.push_front(summary_cache_entry{std::move(key), std::move(value)});
        index_[entries_.front().key] = entries_.begin();

        if (entries_.size() > capacity_) {
            auto last = std::prev(entries_.end());
            index_.erase(last->key);
            entries_.pop_back();
        }
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return entries_.size();
    }

private:
    size_t capacity_;
    mutable std::mutex mutex_;
    std::list<summary_cache_entry> entries_;
    std::unordered_map<std::string, std::list<summary_cache_entry>::iterator> index_;
};

struct compact_attempt {
    json body;
    int prompt_tokens = 0;
    bool compacted = false;
    bool cache_hit = false;
    std::string summary_cache_key;
};

static std::function<void(int)> shutdown_handler;
static std::atomic_flag is_terminating = ATOMIC_FLAG_INIT;

static inline void signal_handler(int signal) {
    if (is_terminating.test_and_set()) {
        fprintf(stderr, "Received second interrupt, terminating immediately.\n");
        exit(1);
    }

    shutdown_handler(signal);
}

static server_http_context::handler_t ex_wrapper(server_http_context::handler_t func) {
    return [func = std::move(func)](const server_http_req & req) -> server_http_res_ptr {
        std::string message;
        error_type error;
        try {
            return func(req);
        } catch (const std::invalid_argument & e) {
            error = ERROR_TYPE_INVALID_REQUEST;
            message = e.what();
        } catch (const std::exception & e) {
            error = ERROR_TYPE_SERVER;
            message = e.what();
        } catch (...) {
            error = ERROR_TYPE_SERVER;
            message = "unknown error";
        }

        auto res = std::make_unique<server_http_res>();
        json error_data = format_error_response(message, error);
        res->status = json_value(error_data, "code", 500);
        res->data = safe_json_to_str({{"error", error_data}});
        return res;
    };
}

static std::string trim_copy(std::string value) {
    auto is_space = [](unsigned char c) {
        return std::isspace(c) != 0;
    };

    value.erase(value.begin(), std::find_if(value.begin(), value.end(), [&](unsigned char c) {
        return !is_space(c);
    }));
    value.erase(std::find_if(value.rbegin(), value.rend(), [&](unsigned char c) {
        return !is_space(c);
    }).base(), value.end());
    return value;
}

static std::string normalize_ws(std::string value) {
    value = trim_copy(std::move(value));

    std::string result;
    result.reserve(value.size());

    bool prev_space = false;
    for (unsigned char c : value) {
        if (std::isspace(c) != 0) {
            if (!prev_space) {
                result.push_back(' ');
                prev_space = true;
            }
        } else {
            result.push_back(static_cast<char>(c));
            prev_space = false;
        }
    }

    return trim_copy(std::move(result));
}

static std::string shorten_text(const std::string & text, size_t max_chars) {
    if (max_chars == 0) {
        return {};
    }

    auto normalized = normalize_ws(text);
    if (normalized.size() <= max_chars) {
        return normalized;
    }

    if (max_chars <= 3) {
        return normalized.substr(0, max_chars);
    }

    return normalized.substr(0, max_chars - 3) + "...";
}

static std::string fnv1a_64_hex(const std::string & input) {
    uint64_t hash = 14695981039346656037ull;
    for (unsigned char c : input) {
        hash ^= c;
        hash *= 1099511628211ull;
    }

    char buf[17];
    snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(hash));
    return buf;
}

static std::string join_url_path(const std::string & base_path, const std::string & request_path) {
    std::string lhs = base_path.empty() ? "/" : base_path;
    std::string rhs = request_path.empty() ? "/" : request_path;

    if (lhs.back() == '/' && rhs.front() == '/') {
        return lhs + rhs.substr(1);
    }
    if (lhs.back() != '/' && rhs.front() != '/') {
        return lhs + "/" + rhs;
    }
    return lhs + rhs;
}

static std::string strip_prefix(const std::string & value, const std::string & prefix) {
    if (!prefix.empty() && value.rfind(prefix, 0) == 0) {
        auto stripped = value.substr(prefix.size());
        return stripped.empty() ? "/" : stripped;
    }
    return value.empty() ? "/" : value;
}

static bool is_tool_call_message(const json & msg) {
    return msg.is_object()
        && json_value(msg, "role", std::string()) == "assistant"
        && msg.contains("tool_calls")
        && msg.at("tool_calls").is_array()
        && !msg.at("tool_calls").empty();
}

static bool is_tool_result_message(const json & msg) {
    const auto role = json_value(msg, "role", std::string());
    return role == "tool" || role == "tool_results";
}

static bool is_system_like_message(const json & msg) {
    const auto role = json_value(msg, "role", std::string());
    return role == "system" || role == "developer";
}

static bool is_plain_conversation_message(const json & msg) {
    const auto role = json_value(msg, "role", std::string());
    return (role == "user" || role == "assistant") && !is_tool_call_message(msg);
}

static std::string content_to_text(const json & content) {
    if (content.is_string()) {
        return content.get<std::string>();
    }

    if (content.is_null()) {
        return {};
    }

    if (!content.is_array()) {
        return content.dump();
    }

    std::string text;
    for (const auto & part : content) {
        const auto type = json_value(part, "type", std::string());
        if (type == "text") {
            if (!text.empty()) {
                text += "\n";
            }
            text += json_value(part, "text", std::string());
        } else {
            if (!text.empty()) {
                text += "\n";
            }
            text += "[" + (type.empty() ? std::string("content") : type) + " omitted]";
        }
    }
    return text;
}

static std::string message_to_text(const json & msg) {
    if (!msg.is_object()) {
        return {};
    }
    if (!msg.contains("content")) {
        return {};
    }
    return content_to_text(msg.at("content"));
}

static void append_memory_block(json & msg, const std::string & memory_block) {
    if (!msg.contains("content") || msg.at("content").is_null()) {
        msg["content"] = memory_block;
        return;
    }

    auto & content = msg["content"];
    if (content.is_string()) {
        std::string base = content.get<std::string>();
        if (!base.empty() && base.back() != '\n') {
            base += "\n\n";
        }
        base += memory_block;
        content = std::move(base);
        return;
    }

    if (!content.is_array()) {
        content = memory_block;
        return;
    }

    content.push_back({
        {"type", "text"},
        {"text", std::string("\n\n") + memory_block},
    });
}

static std::vector<std::string> extract_file_refs(const json & messages) {
    static const std::regex file_regex(R"(([A-Za-z0-9_./-]+\.[A-Za-z0-9_-]+))");
    std::set<std::string> unique;

    for (const auto & msg : messages) {
        const auto text = message_to_text(msg);
        for (auto it = std::sregex_iterator(text.begin(), text.end(), file_regex); it != std::sregex_iterator(); ++it) {
            unique.insert(it->str());
            if (unique.size() >= 12) {
                break;
            }
        }
        if (unique.size() >= 12) {
            break;
        }
    }

    return {unique.begin(), unique.end()};
}

static std::string build_tool_outcome_line(const std::unordered_map<std::string, std::string> & tool_names,
                                           const std::string & call_id,
                                           const std::string & outcome,
                                           size_t max_chars) {
    const auto it = tool_names.find(call_id);
    const std::string tool_name = it == tool_names.end() ? std::string("unknown") : it->second;
    return "- call_id " + call_id + " / function " + tool_name + ": " + shorten_text(outcome, max_chars);
}

static std::string summarize_prefix_messages(const json & prefix_messages, int target_tokens, size_t tool_chars) {
    const size_t budget_chars = std::max<size_t>(80, static_cast<size_t>(target_tokens) * 4);

    std::vector<std::string> user_preferences;
    std::vector<std::string> decisions;
    std::vector<std::string> unresolved;
    std::vector<std::string> tool_outcomes;
    std::unordered_map<std::string, std::string> tool_names;

    static const std::regex pref_re(R"(\b(prefer|please|avoid|don't|do not|need|want|keep|use|never|always)\b)", std::regex::icase);
    static const std::regex decision_re(R"(\b(decid|plan|agreed|use|using|set|configured|implemented|fixed|resolved|chosen)\b)", std::regex::icase);

    for (const auto & msg : prefix_messages) {
        if (is_tool_call_message(msg)) {
            for (const auto & tool_call : msg.at("tool_calls")) {
                const auto call_id = json_value(tool_call, "id", std::string());
                const auto fn = json_value(tool_call.value("function", json::object()), "name", std::string("unknown"));
                tool_names[call_id] = fn;

                const auto args = json_value(tool_call.value("function", json::object()), "arguments", std::string());
                if (!call_id.empty()) {
                    tool_outcomes.push_back("- call_id " + call_id + " / function " + fn + ": invoked with " + shorten_text(args, tool_chars));
                }
            }
            continue;
        }

        if (is_tool_result_message(msg)) {
            const auto call_id = json_value(msg, "tool_call_id", std::string("unknown"));
            const auto outcome = message_to_text(msg);
            tool_outcomes.push_back(build_tool_outcome_line(tool_names, call_id, outcome, tool_chars));
            continue;
        }

        const auto role = json_value(msg, "role", std::string());
        const auto snippet = shorten_text(message_to_text(msg), 220);
        if (snippet.empty()) {
            continue;
        }

        if (role == "user") {
            if (std::regex_search(snippet, pref_re)) {
                user_preferences.push_back("- " + snippet);
            }
            unresolved.push_back("- " + snippet);
        } else if (role == "assistant") {
            if (std::regex_search(snippet, decision_re)) {
                decisions.push_back("- " + snippet);
            }
        }
    }

    auto file_refs = extract_file_refs(prefix_messages);

    if (user_preferences.empty()) {
        user_preferences.push_back("- No durable preference was extracted beyond the recent raw tail.");
    }
    if (decisions.empty()) {
        decisions.push_back("- Earlier turns covered intermediate reasoning, implementation choices, and troubleshooting details.");
    }
    if (unresolved.empty()) {
        unresolved.push_back("- Continue from the recent raw messages; older unresolved details were compacted.");
    }
    if (tool_outcomes.empty()) {
        tool_outcomes.push_back("- No older tool activity was preserved beyond the recent raw tail.");
    }

    std::vector<std::pair<std::string, std::vector<std::string>>> sections;
    sections.push_back({"User preferences", user_preferences});
    sections.push_back({"Decisions made", decisions});

    if (!file_refs.empty()) {
        std::vector<std::string> file_lines;
        std::string current = "- ";
        for (size_t i = 0; i < file_refs.size(); ++i) {
            const std::string candidate = (current == "- " ? "" : ", ") + file_refs[i];
            if (current.size() + candidate.size() > 180) {
                file_lines.push_back(current);
                current = "- " + file_refs[i];
            } else {
                current += candidate;
            }
        }
        if (current != "- ") {
            file_lines.push_back(current);
        }
        sections.push_back({"Relevant code/file context", file_lines});
    }

    if (unresolved.size() > 4) {
        unresolved.erase(unresolved.begin(), unresolved.end() - 4);
    }
    sections.push_back({"Unresolved tasks", unresolved});

    if (tool_outcomes.size() > 6) {
        tool_outcomes.erase(tool_outcomes.begin(), tool_outcomes.end() - 6);
    }
    sections.push_back({"Important tool outcomes", tool_outcomes});

    std::string summary =
        "[COMPACTED_CONVERSATION_MEMORY]\n"
        "This is a lossy summary of earlier conversation. Prefer the newer raw messages below if anything conflicts.\n";

    for (const auto & [title, lines] : sections) {
        if (lines.empty()) {
            continue;
        }
        summary += "\n" + title + ":\n";
        for (const auto & line : lines) {
            if (summary.size() + line.size() + 1 > budget_chars) {
                summary += shorten_text(line, std::max<size_t>(40, budget_chars > summary.size() ? budget_chars - summary.size() - 1 : 0));
                summary += "\n";
                summary += "[/COMPACTED_CONVERSATION_MEMORY]";
                return summary;
            }
            summary += line;
            summary += "\n";
        }
    }

    summary += "[/COMPACTED_CONVERSATION_MEMORY]";
    if (summary.size() > budget_chars) {
        const std::string suffix = "\n[/COMPACTED_CONVERSATION_MEMORY]";
        const size_t body_budget = budget_chars > suffix.size() ? budget_chars - suffix.size() : 0;
        summary = shorten_text(summary, body_budget) + suffix;
    }
    return summary;
}

class proxy_service {
public:
    proxy_service(common_params params, proxy_options options)
        : params_(std::move(params))
        , options_(std::move(options))
        , summary_cache_(static_cast<size_t>(options_.summary_cache_capacity)) {}

    void init() {}

    server_http_res_ptr handle_health(const server_http_req &) {
        auto res = std::make_unique<server_http_res>();
        json body = {
            {"ok", true},
            {"backend_base_url", common_http_show_masked_url(common_http_parse_url(options_.backend_base_url))},
            {"compaction_trigger", options_.compaction_trigger},
            {"hard_prompt_cap", options_.hard_prompt_cap},
            {"model_compaction_triggers", options_.model_compaction_triggers},
            {"model_hard_prompt_caps", options_.model_hard_prompt_caps},
            {"recent_raw_tail", options_.recent_raw_tail},
            {"summary_cache", {
                {"entries", summary_cache_.size()},
                {"hits", summary_cache_hits_.load()},
                {"misses", summary_cache_misses_.load()},
            }},
            {"compactions", compactions_.load()},
        };
        res->data = safe_json_to_str(body);
        return res;
    }

    server_http_res_ptr handle_models(const server_http_req & req) {
        auto upstream = forward_request(req, /* is_get */ true, std::nullopt, /* stream_response */ false);
        if (upstream->status >= 400 || options_.model_alias.empty()) {
            return upstream;
        }

        try {
            json body = json::parse(upstream->data);
            rewrite_model_alias(body);
            upstream->data = safe_json_to_str(body);
        } catch (...) {
        }
        return upstream;
    }

    server_http_res_ptr handle_passthrough_post(const server_http_req & req) {
        return forward_request(req, /* is_get */ false, std::nullopt, false);
    }

    server_http_res_ptr handle_passthrough_get(const server_http_req & req) {
        return forward_request(req, /* is_get */ true, std::nullopt, false);
    }

    server_http_res_ptr handle_chat(const server_http_req & req) {
        std::optional<json> request_body;
        try {
            request_body = json::parse(req.body);
        } catch (...) {
            return forward_request(req, /* is_get */ false, std::nullopt, false);
        }

        if (!request_body->is_object() || !request_body->contains("messages")) {
            return forward_request(req, /* is_get */ false, std::nullopt, false);
        }

        json body = *request_body;
        if (!options_.model_alias.empty()) {
            body["model"] = options_.model_alias;
        }
        const std::string model_name = json_value(body, "model", std::string());
        const int compaction_trigger = select_compaction_trigger(model_name);
        const int hard_prompt_cap = select_hard_prompt_cap(model_name);

        const bool stream = json_value(body, "stream", false);
        auto rendered_prompt = render_prompt(body);
        const int rendered_tokens = tokenize_prompt(rendered_prompt, model_name);

        if (compaction_trigger <= 0 || rendered_tokens <= compaction_trigger) {
            auto res = forward_request(req, /* is_get */ false, safe_json_to_str(body), stream);
            res->headers["X-Llama-Proxy-Compacted"] = "0";
            res->headers["X-Llama-Proxy-Compaction-Trigger"] = std::to_string(compaction_trigger);
            res->headers["X-Llama-Proxy-Hard-Prompt-Cap"] = std::to_string(hard_prompt_cap);
            return res;
        }

        auto compacted = compact_request(std::move(body), hard_prompt_cap);
        if (hard_prompt_cap > 0 && compacted.prompt_tokens > hard_prompt_cap) {
            return make_error_response(
                "unable to compact the conversation under the configured prompt cap",
                ERROR_TYPE_EXCEED_CONTEXT_SIZE);
        }

        if (compacted.compacted) {
            compactions_.fetch_add(1);
        }
        auto res = forward_request(req, /* is_get */ false, safe_json_to_str(compacted.body), stream);
        res->headers["X-Llama-Proxy-Compacted"] = compacted.compacted ? "1" : "0";
        res->headers["X-Llama-Proxy-Compaction-Trigger"] = std::to_string(compaction_trigger);
        res->headers["X-Llama-Proxy-Hard-Prompt-Cap"] = std::to_string(hard_prompt_cap);
        return res;
    }

private:
    common_params params_;
    proxy_options options_;
    summary_cache summary_cache_;

    std::atomic<uint64_t> summary_cache_hits_ = 0;
    std::atomic<uint64_t> summary_cache_misses_ = 0;
    std::atomic<uint64_t> compactions_ = 0;

    int select_hard_prompt_cap(const std::string & model_name) const {
        if (!model_name.empty()) {
            auto it = options_.model_hard_prompt_caps.find(model_name);
            if (it != options_.model_hard_prompt_caps.end()) {
                return it->second;
            }
        }
        return options_.hard_prompt_cap;
    }

    int select_compaction_trigger(const std::string & model_name) const {
        if (!model_name.empty()) {
            auto it = options_.model_compaction_triggers.find(model_name);
            if (it != options_.model_compaction_triggers.end()) {
                return it->second;
            }
        }
        return options_.compaction_trigger;
    }

    static server_http_res_ptr make_error_response(const std::string & message, error_type type) {
        auto res = std::make_unique<server_http_res>();
        json error_data = format_error_response(message, type);
        res->status = json_value(error_data, "code", 500);
        res->data = safe_json_to_str({{"error", error_data}});
        return res;
    }

    void rewrite_model_alias(json & body) const {
        if (body.contains("data") && body.at("data").is_array() && !body.at("data").empty()) {
            for (auto & model : body["data"]) {
                model["id"] = options_.model_alias;
            }
        }

        if (body.contains("models") && body.at("models").is_array() && !body.at("models").empty()) {
            for (auto & model : body["models"]) {
                model["name"] = options_.model_alias;
                model["model"] = options_.model_alias;
            }
        }
    }

    std::pair<httplib::Client, common_http_url> make_backend_client() const {
        auto [client, parts] = common_http_client(options_.backend_base_url);
        client.set_read_timeout(params_.timeout_read);
        client.set_write_timeout(params_.timeout_write);
        return {std::move(client), std::move(parts)};
    }

    static httplib::Headers copy_request_headers(const std::map<std::string, std::string> & headers,
                                                 const std::string & backend_api_key) {
        static const std::unordered_set<std::string> skip = {
            "authorization",
            "connection",
            "content-length",
            "host",
            "transfer-encoding",
            "x-api-key",
        };

        httplib::Headers result;
        for (const auto & [key, value] : headers) {
            std::string lowered = key;
            std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
            if (skip.count(lowered) != 0) {
                continue;
            }
            result.emplace(key, value);
        }

        if (!backend_api_key.empty()) {
            result.emplace("Authorization", "Bearer " + backend_api_key);
        }
        return result;
    }

    static void apply_response_headers(server_http_res & dst, const httplib::Headers & headers) {
        static const std::unordered_set<std::string> skip = {
            "connection",
            "content-length",
            "transfer-encoding",
        };

        for (const auto & [key, value] : headers) {
            std::string lowered = key;
            std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
            if (skip.count(lowered) != 0) {
                continue;
            }
            if (lowered == "content-type") {
                dst.content_type = value;
            } else {
                dst.headers[key] = value;
            }
        }
    }

    server_http_res_ptr forward_request(const server_http_req & req,
                                        bool is_get,
                                        const std::optional<std::string> & body_override,
                                        bool stream_response) const {
        auto [client, parts] = make_backend_client();
        const auto upstream_path = strip_prefix(req.path, params_.api_prefix);
        std::string path = join_url_path(parts.path, upstream_path);
        if (!req.query_string.empty()) {
            path += "?" + req.query_string;
        }

        const auto headers = copy_request_headers(req.headers, options_.backend_api_key);

        if (is_get && stream_response) {
            stream_response = false;
        }

        if (is_get) {
            if (const auto result = client.Get(path.c_str(), headers); result) {
                auto res = std::make_unique<server_http_res>();
                res->status = result->status;
                res->data = result->body;
                apply_response_headers(*res, result->headers);
                return res;
            } else {
                return make_error_response("backend GET failed: " + httplib::to_string(result.error()), ERROR_TYPE_UNAVAILABLE);
            }
        }

        const std::string body = body_override.has_value() ? *body_override : req.body;
        const std::string content_type = [&]() -> std::string {
            auto it = req.headers.find("Content-Type");
            if (it != req.headers.end()) {
                return it->second;
            }
            return "application/json; charset=utf-8";
        }();

        if (stream_response) {
            auto stream = std::make_shared<httplib::stream::Result>(
                httplib::stream::Post(client, path.c_str(), headers, body, content_type));

            if (!stream->is_valid()) {
                return make_error_response("backend streaming POST failed: " + httplib::to_string(stream->error()), ERROR_TYPE_UNAVAILABLE);
            }

            auto res = std::make_unique<server_http_res>();
            res->status = stream->status();
            apply_response_headers(*res, stream->headers());
            res->next = [stream](std::string & chunk) mutable -> bool {
                if (!stream->next()) {
                    chunk.clear();
                    return false;
                }

                chunk.assign(stream->data(), stream->size());
                return true;
            };
            return res;
        }

        if (const auto result = client.Post(path.c_str(), headers, body, content_type); result) {
            auto res = std::make_unique<server_http_res>();
            res->status = result->status;
            res->data = result->body;
            apply_response_headers(*res, result->headers);
            return res;
        } else {
            return make_error_response("backend POST failed: " + httplib::to_string(result.error()), ERROR_TYPE_UNAVAILABLE);
        }
    }

    std::string render_prompt(const json & body) const {
        auto [client, parts] = make_backend_client();
        const auto path = join_url_path(parts.path, "/apply-template");
        const auto headers = copy_request_headers({}, options_.backend_api_key);
        auto result = client.Post(path.c_str(), headers, safe_json_to_str(body), "application/json; charset=utf-8");
        if (!result) {
            throw std::runtime_error("backend /apply-template failed: " + httplib::to_string(result.error()));
        }
        if (result->status >= 400) {
            throw std::runtime_error("backend /apply-template returned HTTP " + std::to_string(result->status) + ": " + result->body);
        }

        json rendered = json::parse(result->body);
        return rendered.at("prompt").get<std::string>();
    }

    int tokenize_prompt(const std::string & prompt, const std::string & model_name = {}) const {
        auto [client, parts] = make_backend_client();
        const auto path = join_url_path(parts.path, "/tokenize");
        const auto headers = copy_request_headers({}, options_.backend_api_key);
        json body = {
            {"content", prompt},
            {"add_special", false},
            {"parse_special", true},
        };
        if (!model_name.empty()) {
            body["model"] = model_name;
        }

        auto result = client.Post(path.c_str(), headers, safe_json_to_str(body), "application/json; charset=utf-8");
        if (!result) {
            throw std::runtime_error("backend /tokenize failed: " + httplib::to_string(result.error()));
        }
        if (result->status >= 400) {
            throw std::runtime_error("backend /tokenize returned HTTP " + std::to_string(result->status) + ": " + result->body);
        }

        const json parsed = json::parse(result->body);
        return static_cast<int>(parsed.at("tokens").size());
    }

    std::string get_cached_or_build_summary(const json & prefix_messages,
                                            int summary_target,
                                            size_t tool_chars,
                                            bool & cache_hit,
                                            std::string & cache_key) {
        cache_key = fnv1a_64_hex(prefix_messages.dump() + "|" + std::to_string(summary_target) + "|" + std::to_string(tool_chars));

        std::string summary;
        if (summary_cache_.get(cache_key, summary)) {
            cache_hit = true;
            summary_cache_hits_.fetch_add(1);
            return summary;
        }

        cache_hit = false;
        summary_cache_misses_.fetch_add(1);
        summary = summarize_prefix_messages(prefix_messages, summary_target, tool_chars);
        summary_cache_.put(cache_key, summary);
        return summary;
    }

    static size_t count_leading_system_messages(const json & messages) {
        size_t count = 0;
        while (count < messages.size() && is_system_like_message(messages.at(count))) {
            ++count;
        }
        return count;
    }

    static size_t find_tail_boundary(const json & messages, size_t system_prefix_count, int tail_plain_messages) {
        int seen_plain = 0;
        std::optional<size_t> earliest_plain;

        for (size_t i = messages.size(); i > system_prefix_count; --i) {
            const size_t idx = i - 1;
            if (!is_plain_conversation_message(messages.at(idx))) {
                continue;
            }
            earliest_plain = idx;
            ++seen_plain;
            if (seen_plain >= tail_plain_messages) {
                break;
            }
        }

        if (earliest_plain.has_value()) {
            return *earliest_plain;
        }

        if (messages.size() <= system_prefix_count) {
            return system_prefix_count;
        }

        const size_t fallback_keep = 4;
        return std::max(system_prefix_count, messages.size() > fallback_keep ? messages.size() - fallback_keep : system_prefix_count);
    }

    static json rebuild_messages_with_summary(const json & original_messages,
                                              size_t system_prefix_count,
                                              size_t tail_boundary,
                                              const std::string & summary_block) {
        json rebuilt = json::array();

        for (size_t i = 0; i < system_prefix_count; ++i) {
            rebuilt.push_back(original_messages.at(i));
        }

        if (!summary_block.empty()) {
            if (!rebuilt.empty()) {
                append_memory_block(rebuilt.at(0), summary_block);
            } else {
                rebuilt.push_back({
                    {"role", "system"},
                    {"content", summary_block},
                });
            }
        }

        for (size_t i = tail_boundary; i < original_messages.size(); ++i) {
            rebuilt.push_back(original_messages.at(i));
        }

        return rebuilt;
    }

    compact_attempt compact_request(json body, int hard_prompt_cap) {
        compact_attempt best;
        best.body = body;
        const std::string model_name = json_value(body, "model", std::string());

        if (!body.contains("messages") || !body.at("messages").is_array()) {
            best.prompt_tokens = tokenize_prompt(render_prompt(body), model_name);
            return best;
        }

        const json original_messages = body.at("messages");
        const size_t system_prefix_count = count_leading_system_messages(original_messages);

        std::vector<int> summary_targets = {options_.first_summary_target, options_.second_summary_target};
        std::vector<int> tail_candidates = {options_.recent_raw_tail};
        if (options_.recent_raw_tail > 6) {
            tail_candidates.push_back(6);
        }
        if (options_.recent_raw_tail > 4) {
            tail_candidates.push_back(4);
        }
        std::sort(summary_targets.begin(), summary_targets.end(), std::greater<int>());
        summary_targets.erase(std::unique(summary_targets.begin(), summary_targets.end()), summary_targets.end());

        int best_tokens = std::numeric_limits<int>::max();

        for (int summary_target : summary_targets) {
            for (size_t tool_chars : {size_t(200), size_t(120), size_t(80)}) {
                for (int tail_n : tail_candidates) {
                    const size_t boundary = find_tail_boundary(original_messages, system_prefix_count, tail_n);
                    if (boundary <= system_prefix_count) {
                        continue;
                    }

                    json prefix_messages = json::array();
                    for (size_t i = system_prefix_count; i < boundary; ++i) {
                        prefix_messages.push_back(original_messages.at(i));
                    }

                    bool cache_hit = false;
                    std::string cache_key;
                    const std::string summary = get_cached_or_build_summary(prefix_messages, summary_target, tool_chars, cache_hit, cache_key);

                    json candidate = body;
                    candidate["messages"] = rebuild_messages_with_summary(original_messages, system_prefix_count, boundary, summary);

                    const int candidate_tokens = tokenize_prompt(render_prompt(candidate), model_name);
                    if (candidate_tokens < best_tokens) {
                        best_tokens = candidate_tokens;
                        best.body = candidate;
                        best.prompt_tokens = candidate_tokens;
                        best.compacted = true;
                        best.cache_hit = cache_hit;
                        best.summary_cache_key = cache_key;
                    }

                    if (hard_prompt_cap > 0 && candidate_tokens <= hard_prompt_cap) {
                        return best;
                    }
                }
            }
        }

        if (!best.compacted) {
            best.prompt_tokens = tokenize_prompt(render_prompt(body), model_name);
        }
        return best;
    }
};

static void print_proxy_usage(char ** argv) {
    LOG("\nusage: %s [proxy options] [llama-server HTTP options]\n", argv[0]);
    LOG("\nproxy options:\n");
    LOG("  --backend-base-url URL         backend llama-server base URL (default: http://127.0.0.1:8080)\n");
    LOG("  --backend-api-key KEY          backend Authorization bearer token\n");
    LOG("  --chat-template-file PATH      deprecated, ignored (proxy uses backend /apply-template)\n");
    LOG("  --model-alias NAME             model alias to inject into proxied chat requests\n");
    LOG("  --hard-prompt-cap N            max rendered prompt tokens after compaction; 0 disables the cap (default: 32000)\n");
    LOG("  --compaction-trigger N         trigger compaction above this rendered token count; 0 disables compaction (default: 24000)\n");
    LOG("  --model-hard-prompt-cap M=N    per-model prompt cap override, repeatable\n");
    LOG("  --model-compaction-trigger M=N per-model compaction trigger override, repeatable\n");
    LOG("  --recent-raw-tail N            keep the latest N plain chat turns verbatim (default: 8)\n");
    LOG("  --first-summary-target N       first-pass summary target in tokens (default: 800)\n");
    LOG("  --second-summary-target N      second-pass summary target in tokens (default: 400)\n");
    LOG("  --summary-cache-capacity N     in-memory summary cache size (default: 128)\n");
    LOG("\nllama-server HTTP options still supported here include: --host, --port, --api-key, --api-prefix,\n");
    LOG("--timeout-read, --timeout-write, --ssl-key-file, --ssl-cert-file, and related listener settings.\n\n");
}

static bool parse_proxy_args(int argc,
                             char ** argv,
                             proxy_options & options,
                             std::vector<char *> & remaining_args) {
    remaining_args.clear();
    remaining_args.push_back(argv[0]);

    auto require_value = [&](int & index, const char * arg_name) -> const char * {
        if (index + 1 >= argc) {
            throw std::invalid_argument(std::string("missing value for ") + arg_name);
        }
        ++index;
        return argv[index];
    };

    auto parse_model_int = [](const std::string & raw, const char * arg_name) -> std::pair<std::string, int> {
        const auto pos = raw.find('=');
        if (pos == std::string::npos || pos == 0 || pos + 1 >= raw.size()) {
            throw std::invalid_argument(std::string("expected MODEL=VALUE for ") + arg_name);
        }

        const std::string model = trim_copy(raw.substr(0, pos));
        const std::string value_str = trim_copy(raw.substr(pos + 1));
        if (model.empty() || value_str.empty()) {
            throw std::invalid_argument(std::string("expected MODEL=VALUE for ") + arg_name);
        }

        return {model, std::stoi(value_str)};
    };

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_proxy_usage(argv);
            return false;
        }
        if (arg == "--backend-base-url") {
            options.backend_base_url = require_value(i, "--backend-base-url");
            continue;
        }
        if (arg == "--backend-api-key") {
            options.backend_api_key = require_value(i, "--backend-api-key");
            continue;
        }
        if (arg == "--chat-template-file") {
            const char * ignored = require_value(i, "--chat-template-file");
            LOG_WRN("%s: ignoring deprecated --chat-template-file=%s, proxy now renders via backend /apply-template\n", __func__, ignored);
            continue;
        }
        if (arg == "--model-alias") {
            options.model_alias = require_value(i, "--model-alias");
            continue;
        }
        if (arg == "--hard-prompt-cap") {
            options.hard_prompt_cap = std::stoi(require_value(i, "--hard-prompt-cap"));
            continue;
        }
        if (arg == "--compaction-trigger") {
            options.compaction_trigger = std::stoi(require_value(i, "--compaction-trigger"));
            continue;
        }
        if (arg == "--model-hard-prompt-cap") {
            auto [model, value] = parse_model_int(require_value(i, "--model-hard-prompt-cap"), "--model-hard-prompt-cap");
            options.model_hard_prompt_caps[model] = value;
            continue;
        }
        if (arg == "--model-compaction-trigger") {
            auto [model, value] = parse_model_int(require_value(i, "--model-compaction-trigger"), "--model-compaction-trigger");
            options.model_compaction_triggers[model] = value;
            continue;
        }
        if (arg == "--recent-raw-tail") {
            options.recent_raw_tail = std::stoi(require_value(i, "--recent-raw-tail"));
            continue;
        }
        if (arg == "--first-summary-target") {
            options.first_summary_target = std::stoi(require_value(i, "--first-summary-target"));
            continue;
        }
        if (arg == "--second-summary-target") {
            options.second_summary_target = std::stoi(require_value(i, "--second-summary-target"));
            continue;
        }
        if (arg == "--summary-cache-capacity") {
            options.summary_cache_capacity = std::stoi(require_value(i, "--summary-cache-capacity"));
            continue;
        }

        remaining_args.push_back(argv[i]);
    }

    return true;
}

} // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    proxy_options options;
    std::vector<char *> remaining_args;
    if (!parse_proxy_args(argc, argv, options, remaining_args)) {
        return 0;
    }

    common_params params;
    const int remaining_argc = static_cast<int>(remaining_args.size());
    if (!common_params_parse(remaining_argc, remaining_args.data(), params, LLAMA_EXAMPLE_SERVER)) {
        return 1;
    }

    common_init();

    proxy_service proxy(params, options);
    proxy.init();

    server_http_context ctx_http;
    if (!ctx_http.init(params)) {
        LOG_ERR("%s: failed to initialize HTTP server\n", __func__);
        return 1;
    }

    ctx_http.get ("/health",              ex_wrapper([&](const server_http_req & req) { return proxy.handle_health(req); }));
    ctx_http.get ("/v1/health",           ex_wrapper([&](const server_http_req & req) { return proxy.handle_health(req); }));
    ctx_http.get ("/metrics",             ex_wrapper([&](const server_http_req & req) { return proxy.handle_passthrough_get(req); }));
    ctx_http.get ("/props",               ex_wrapper([&](const server_http_req & req) { return proxy.handle_passthrough_get(req); }));
    ctx_http.post("/props",               ex_wrapper([&](const server_http_req & req) { return proxy.handle_passthrough_post(req); }));
    ctx_http.post("/api/show",            ex_wrapper([&](const server_http_req & req) { return proxy.handle_passthrough_post(req); }));
    ctx_http.get ("/models",              ex_wrapper([&](const server_http_req & req) { return proxy.handle_models(req); }));
    ctx_http.get ("/v1/models",           ex_wrapper([&](const server_http_req & req) { return proxy.handle_models(req); }));
    ctx_http.get ("/api/tags",            ex_wrapper([&](const server_http_req & req) { return proxy.handle_models(req); }));
    ctx_http.post("/completion",          ex_wrapper([&](const server_http_req & req) { return proxy.handle_passthrough_post(req); }));
    ctx_http.post("/completions",         ex_wrapper([&](const server_http_req & req) { return proxy.handle_passthrough_post(req); }));
    ctx_http.post("/v1/completions",      ex_wrapper([&](const server_http_req & req) { return proxy.handle_passthrough_post(req); }));
    ctx_http.post("/chat/completions",    ex_wrapper([&](const server_http_req & req) { return proxy.handle_chat(req); }));
    ctx_http.post("/v1/chat/completions", ex_wrapper([&](const server_http_req & req) { return proxy.handle_chat(req); }));
    ctx_http.post("/api/chat",            ex_wrapper([&](const server_http_req & req) { return proxy.handle_chat(req); }));
    ctx_http.post("/v1/responses",        ex_wrapper([&](const server_http_req & req) { return proxy.handle_passthrough_post(req); }));
    ctx_http.post("/responses",           ex_wrapper([&](const server_http_req & req) { return proxy.handle_passthrough_post(req); }));
    ctx_http.post("/v1/messages",         ex_wrapper([&](const server_http_req & req) { return proxy.handle_passthrough_post(req); }));
    ctx_http.post("/v1/messages/count_tokens", ex_wrapper([&](const server_http_req & req) { return proxy.handle_passthrough_post(req); }));
    ctx_http.post("/infill",              ex_wrapper([&](const server_http_req & req) { return proxy.handle_passthrough_post(req); }));
    ctx_http.post("/embedding",           ex_wrapper([&](const server_http_req & req) { return proxy.handle_passthrough_post(req); }));
    ctx_http.post("/embeddings",          ex_wrapper([&](const server_http_req & req) { return proxy.handle_passthrough_post(req); }));
    ctx_http.post("/v1/embeddings",       ex_wrapper([&](const server_http_req & req) { return proxy.handle_passthrough_post(req); }));
    ctx_http.post("/rerank",              ex_wrapper([&](const server_http_req & req) { return proxy.handle_passthrough_post(req); }));
    ctx_http.post("/reranking",           ex_wrapper([&](const server_http_req & req) { return proxy.handle_passthrough_post(req); }));
    ctx_http.post("/v1/rerank",           ex_wrapper([&](const server_http_req & req) { return proxy.handle_passthrough_post(req); }));
    ctx_http.post("/v1/reranking",        ex_wrapper([&](const server_http_req & req) { return proxy.handle_passthrough_post(req); }));
    ctx_http.post("/tokenize",            ex_wrapper([&](const server_http_req & req) { return proxy.handle_passthrough_post(req); }));
    ctx_http.post("/detokenize",          ex_wrapper([&](const server_http_req & req) { return proxy.handle_passthrough_post(req); }));
    ctx_http.post("/apply-template",      ex_wrapper([&](const server_http_req & req) { return proxy.handle_passthrough_post(req); }));
    ctx_http.get ("/lora-adapters",       ex_wrapper([&](const server_http_req & req) { return proxy.handle_passthrough_get(req); }));
    ctx_http.post("/lora-adapters",       ex_wrapper([&](const server_http_req & req) { return proxy.handle_passthrough_post(req); }));
    ctx_http.get ("/slots",               ex_wrapper([&](const server_http_req & req) { return proxy.handle_passthrough_get(req); }));
    ctx_http.post("/slots/:id_slot",      ex_wrapper([&](const server_http_req & req) { return proxy.handle_passthrough_post(req); }));
    if (params.webui_mcp_proxy) {
        LOG_WRN("%s", "-----------------\n");
        LOG_WRN("%s", "CORS proxy is enabled, do not expose server to untrusted environments\n");
        LOG_WRN("%s", "This feature is EXPERIMENTAL and may be removed or changed in future versions\n");
        LOG_WRN("%s", "-----------------\n");
        ctx_http.get ("/cors-proxy",      ex_wrapper(proxy_handler_get));
        ctx_http.post("/cors-proxy",      ex_wrapper(proxy_handler_post));
    }

    if (!ctx_http.start()) {
        LOG_ERR("%s: failed to start HTTP server\n", __func__);
        return 1;
    }
    ctx_http.is_ready.store(true);

    shutdown_handler = [&](int) {
        ctx_http.stop();
    };

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__)) || defined(_WIN32)
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
#endif

    if (ctx_http.thread.joinable()) {
        ctx_http.thread.join();
    }

    return 0;
}
