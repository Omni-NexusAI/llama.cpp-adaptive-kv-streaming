#include "../src/llama-kv-stream-group.h"
#include "testing.h"

#include <stdexcept>
#include <string>

// Records calls and fails on request. Tokens advance by `step` on a successful complete().
struct mock_member : llama_kv_stream_member {
    std::vector<std::string> * log;
    std::string name;
    size_t count = 0, step = 8;
    std::string fail_op;
    bool throw_on_fail = false;
    int aborts = 0;
    mock_member(std::vector<std::string> * l, std::string n) : log(l), name(std::move(n)) {}
    // Returns true when this call must fail.
    bool injected(const char * op) {
        log->push_back(name + ":" + op);
        if (fail_op == op) {
            if (throw_on_fail) throw std::runtime_error("injected");
            return true;
        }
        return false;
    }
    bool begin(size_t, uint32_t, bool) override { return !injected("begin"); }
    bool complete() override {
        if (injected("complete")) return false;
        count += step;
        return true;
    }
    void abort() override { log->push_back(name + ":abort"); ++aborts; }
    bool truncate(size_t n) override { if (injected("truncate")) return false; count = n; return true; }
    bool restore(size_t n) override { if (injected("restore")) return false; count = n; return true; }
    bool reset(bool) override { if (injected("reset")) return false; count = 0; return true; }
    size_t tokens() const override { return count; }
};

struct fixture {
    std::vector<std::string> log;
    mock_member * a = nullptr, * b = nullptr;
    std::unique_ptr<llama_kv_stream_group> group;
    fixture() {
        auto ma = std::make_unique<mock_member>(&log, "A");
        auto mb = std::make_unique<mock_member>(&log, "B");
        a = ma.get();
        b = mb.get();
        std::vector<std::unique_ptr<llama_kv_stream_member>> v;
        v.push_back(std::move(ma));
        v.push_back(std::move(mb));
        group = std::make_unique<llama_kv_stream_group>(std::move(v));
    }
    size_t count_calls(const std::string & what) const {
        size_t n = 0;
        for (auto & s : log) if (s == what) ++n;
        return n;
    }
};

int main() {
    testing t;
    t.test("append_cycle_advances_every_member_to_one_frontier", [](testing & t) {
        fixture f;
        t.assert_true(f.group->begin(8, 8, false));
        t.assert_true(f.group->active());
        t.assert_true(f.group->complete());
        t.assert_equal(size_t(8), f.group->tokens());
        t.assert_equal(size_t(8), f.a->tokens());
        t.assert_equal(size_t(8), f.b->tokens());
        t.assert_true(!f.group->active() && !f.group->failed());
    });
    t.test("begin_failure_on_second_member_aborts_both_and_fails_group", [](testing & t) {
        fixture f;
        f.b->fail_op = "begin";
        t.assert_true(!f.group->begin(8, 8, false));
        t.assert_true(f.group->failed() && !f.group->active());
        t.assert_equal(1, f.a->aborts);
        t.assert_equal(1, f.b->aborts);
        t.assert_true(!f.group->begin(8, 8, false));
    });
    t.test("complete_failure_still_calls_every_member_then_aborts_all", [](testing & t) {
        fixture f;
        t.assert_true(f.group->begin(8, 8, false));
        f.a->fail_op = "complete";
        t.assert_true(!f.group->complete());
        t.assert_equal(size_t(1), f.count_calls("A:complete"));
        t.assert_equal(size_t(1), f.count_calls("B:complete"));
        t.assert_true(f.group->failed());
        t.assert_equal(1, f.a->aborts);
        t.assert_equal(1, f.b->aborts);
        t.assert_equal(size_t(0), f.group->tokens());
    });
    t.test("exception_in_a_member_is_a_failure_not_a_crash", [](testing & t) {
        fixture f;
        f.b->fail_op = "begin";
        f.b->throw_on_fail = true;
        t.assert_true(!f.group->begin(8, 8, false));
        t.assert_true(f.group->failed());
        t.assert_equal(1, f.a->aborts);
    });
    t.test("divergent_frontier_after_complete_is_detected", [](testing & t) {
        fixture f;
        f.b->step = 4;
        t.assert_true(f.group->begin(8, 8, false));
        t.assert_true(!f.group->complete());
        t.assert_true(f.group->failed());
    });
    t.test("truncate_applies_to_all_or_fails_group", [](testing & t) {
        fixture f;
        t.assert_true(f.group->begin(8, 8, false) && f.group->complete());
        t.assert_true(f.group->begin(8, 8, false) && f.group->complete());
        t.assert_true(!f.group->truncate(99));
        t.assert_true(!f.group->failed());
        t.assert_true(f.group->truncate(8));
        t.assert_equal(size_t(8), f.a->tokens());
        t.assert_equal(size_t(8), f.b->tokens());
        f.b->fail_op = "truncate";
        t.assert_true(!f.group->truncate(0));
        t.assert_true(f.group->failed());
    });
    t.test("reconstruct_restores_a_failed_group_to_a_common_frontier", [](testing & t) {
        fixture f;
        f.b->fail_op = "begin";
        t.assert_true(!f.group->begin(8, 8, false));
        f.b->fail_op.clear();
        t.assert_true(f.group->reconstruct(0));
        t.assert_true(!f.group->failed());
        t.assert_true(f.group->begin(8, 8, false) && f.group->complete());
        t.assert_equal(size_t(8), f.group->tokens());
    });
    t.test("reconstruct_fails_closed_when_a_member_cannot_restore", [](testing & t) {
        fixture f;
        f.group->abort();
        f.a->fail_op = "restore";
        t.assert_true(!f.group->reconstruct(0));
        t.assert_true(f.group->failed());
    });
    t.test("mismatched_members_at_construction_start_failed", [](testing & t) {
        std::vector<std::string> log;
        auto ma = std::make_unique<mock_member>(&log, "A");
        auto mb = std::make_unique<mock_member>(&log, "B");
        mb->count = 256;
        std::vector<std::unique_ptr<llama_kv_stream_member>> v;
        v.push_back(std::move(ma));
        v.push_back(std::move(mb));
        llama_kv_stream_group g(std::move(v));
        t.assert_true(g.failed() && !g.consistent());
        t.assert_true(!g.begin(8, 8, false));
    });
    t.test("complete_without_begin_and_double_begin_are_refused", [](testing & t) {
        fixture f;
        t.assert_true(!f.group->complete());
        t.assert_true(f.group->begin(8, 8, false));
        t.assert_true(!f.group->begin(8, 8, false));
    });
    return t.summary();
}
