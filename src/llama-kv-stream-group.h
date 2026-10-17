#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

// One device's serial KV stream session, seen through the calls the group needs.
// llama_kv_stream_model is adapted to this interface; tests use mocks.
class llama_kv_stream_member {
public:
    virtual ~llama_kv_stream_member() = default;
    virtual bool begin(size_t active_tokens, uint32_t query_tokens, bool decode) = 0;
    virtual bool complete() = 0;
    virtual void abort() = 0;
    virtual bool truncate(size_t tokens) = 0;
    virtual bool restore(size_t tokens) = 0;
    virtual bool reset(bool clear_bytes) = 0;
    virtual size_t tokens() const = 0;
};

class llama_kv_stream_model;
// Wrap one per-device model. The model is borrowed and must outlive the member.
std::unique_ptr<llama_kv_stream_member> llama_kv_stream_member_from_model(llama_kv_stream_model * model);

// All-or-nothing coordinator over per-device members that share one logical token frontier.
// Any member failure, exception, or frontier mismatch aborts every member and marks the group failed.
// A failed group refuses work until reconstruct() restores every member to a chosen frontier.
// Owner-thread only. Members are called in device order. Device work is already queued on separate streams,
// so waiting on members one after another costs the slowest member, not the sum.
class llama_kv_stream_group {
public:
    explicit llama_kv_stream_group(std::vector<std::unique_ptr<llama_kv_stream_member>> members);

    bool begin(size_t active_tokens, uint32_t query_tokens, bool decode);
    bool complete();
    void abort();
    bool truncate(size_t tokens);
    bool restore(size_t tokens);
    bool reset(bool clear_bytes);
    bool reconstruct(size_t tokens);

    size_t tokens() const noexcept { return frontier; }
    size_t size() const noexcept { return members.size(); }
    bool active() const noexcept { return in_flight; }
    bool failed() const noexcept { return broken; }
    // True when all members report the same token count.
    bool consistent() const;

private:
    void fail();
    template<typename F> bool each(F && op);

    std::vector<std::unique_ptr<llama_kv_stream_member>> members;
    size_t frontier = 0;
    bool in_flight = false;
    bool broken = false;
};
