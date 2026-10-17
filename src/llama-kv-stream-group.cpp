#include "llama-kv-stream-group.h"
#include "llama-kv-stream-model.h"

#include <exception>

llama_kv_stream_group::llama_kv_stream_group(std::vector<std::unique_ptr<llama_kv_stream_member>> m) :
        members(std::move(m)) {
    if (members.empty()) {
        broken = true;
        return;
    }
    frontier = members.front()->tokens();
    if (!consistent()) broken = true;
}

bool llama_kv_stream_group::consistent() const {
    for (const auto & m : members) {
        if (m->tokens() != members.front()->tokens()) return false;
    }
    return !members.empty();
}

// Run op on every member, even after a failure, so each member sees the same call sequence.
template<typename F>
bool llama_kv_stream_group::each(F && op) {
    bool ok = true;
    for (auto & m : members) {
        bool result = false;
        try {
            result = op(*m);
        } catch (...) {
            result = false;
        }
        ok = result && ok;
    }
    return ok;
}

void llama_kv_stream_group::fail() {
    each([](llama_kv_stream_member & m) { m.abort(); return true; });
    in_flight = false;
    broken = true;
}

bool llama_kv_stream_group::begin(size_t active_tokens, uint32_t query_tokens, bool decode) {
    if (broken || in_flight) return false;
    if (!consistent()) {
        fail();
        return false;
    }
    for (auto & m : members) {
        bool ok = false;
        try {
            ok = m->begin(active_tokens, query_tokens, decode);
        } catch (...) {
            ok = false;
        }
        if (!ok) {
            fail();
            return false;
        }
    }
    in_flight = true;
    return true;
}

bool llama_kv_stream_group::complete() {
    if (broken || !in_flight) return false;
    const bool ok = each([](llama_kv_stream_member & m) { return m.complete(); });
    if (!ok || !consistent()) {
        fail();
        return false;
    }
    in_flight = false;
    frontier = members.front()->tokens();
    return true;
}

void llama_kv_stream_group::abort() {
    fail();
}

bool llama_kv_stream_group::truncate(size_t tokens) {
    if (broken || in_flight || tokens > frontier) return false;
    const bool ok = each([&](llama_kv_stream_member & m) { return m.truncate(tokens); });
    if (!ok || !consistent() || members.front()->tokens() != tokens) {
        fail();
        return false;
    }
    frontier = tokens;
    return true;
}

bool llama_kv_stream_group::restore(size_t tokens) {
    if (broken || in_flight) return false;
    const bool ok = each([&](llama_kv_stream_member & m) { return m.restore(tokens); });
    if (!ok || !consistent() || members.front()->tokens() != tokens) {
        fail();
        return false;
    }
    frontier = tokens;
    return true;
}

bool llama_kv_stream_group::reset(bool clear_bytes) {
    if (broken || in_flight) return false;
    const bool ok = each([&](llama_kv_stream_member & m) { return m.reset(clear_bytes); });
    if (!ok || !consistent() || members.front()->tokens() != 0) {
        fail();
        return false;
    }
    frontier = 0;
    return true;
}

bool llama_kv_stream_group::reconstruct(size_t tokens) {
    if (members.empty()) return false;
    each([](llama_kv_stream_member & m) { m.abort(); return true; });
    in_flight = false;
    const bool ok = each([&](llama_kv_stream_member & m) { return m.restore(tokens); });
    if (!ok || !consistent() || members.front()->tokens() != tokens) {
        broken = true;
        return false;
    }
    frontier = tokens;
    broken = false;
    return true;
}

namespace {
struct model_member : llama_kv_stream_member {
    llama_kv_stream_model * model;
    explicit model_member(llama_kv_stream_model * m) : model(m) {}
    bool begin(size_t active_tokens, uint32_t query_tokens, bool decode) override { return model->begin(active_tokens, query_tokens, decode); }
    bool complete() override { return model->complete(); }
    void abort() override { model->abort(); }
    bool truncate(size_t tokens) override { return model->truncate(tokens); }
    bool restore(size_t tokens) override { return model->restore(tokens); }
    bool reset(bool clear_bytes) override { return model->reset(clear_bytes); }
    size_t tokens() const override { return model->tokens(); }
};
}

std::unique_ptr<llama_kv_stream_member> llama_kv_stream_member_from_model(llama_kv_stream_model * model) {
    if (!model) return nullptr;
    return std::make_unique<model_member>(model);
}
