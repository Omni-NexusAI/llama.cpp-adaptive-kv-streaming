#include "../src/llama-kv-stream-shard-map.h"
#include "testing.h"

// Qwen3.8-27B family: 24 Q heads, 4 KV heads, head dim 256. Q8_0 and Q4_0 blocks are 32 elements.
static llama_kv_stream_shard_params qwen(std::vector<uint32_t> heads) {
    llama_kv_stream_shard_params p;
    p.n_head = 24; p.n_head_kv = 4; p.head_dim_k = 256; p.head_dim_v = 256; p.block_k = 32; p.block_v = 32;
    p.kv_heads = std::move(heads);
    return p;
}

int main() {
    testing t;
    t.test("stock_rotation_alternates_one_three_and_two_two_at_ts_35_65", [](testing & t) {
        const std::vector<float> ts = {0.35f, 0.65f};
        t.assert_true(std::vector<uint32_t>({1, 3}) == llama_kv_stream_stock_kv_heads(4, 256, ts, 0));
        t.assert_true(std::vector<uint32_t>({2, 2}) == llama_kv_stream_stock_kv_heads(4, 256, ts, 1));
        uint32_t dev0 = 0, dev1 = 0;
        for (uint32_t layer = 0; layer < 16; ++layer) {
            auto h = llama_kv_stream_stock_kv_heads(4, 256, ts, layer % 2);
            dev0 += h[0]; dev1 += h[1];
        }
        t.assert_equal(uint32_t(24), dev0);
        t.assert_equal(uint32_t(40), dev1);
    });
    t.test("stock_is_uniform_only_at_exact_quarter_or_half", [](testing & t) {
        for (auto ts : {std::vector<float>{0.25f, 0.75f}, std::vector<float>{0.5f, 0.5f}}) {
            auto a = llama_kv_stream_stock_kv_heads(4, 256, ts, 0), b = llama_kv_stream_stock_kv_heads(4, 256, ts, 1);
            t.assert_true(a == b);
        }
    });
    t.test("map_assigns_contiguous_whole_gqa_groups", [](testing & t) {
        std::vector<llama_kv_stream_shard> out;
        auto r = llama_kv_stream_shard_map_make(qwen({3, 1}), out);
        t.assert_true(r.status == llama_kv_stream_shard_status::success);
        t.assert_equal(size_t(2), out.size());
        t.assert_equal(uint32_t(0), out[0].kv_head_begin);
        t.assert_equal(uint32_t(3), out[0].kv_head_count);
        t.assert_equal(uint32_t(18), out[0].q_head_count);
        t.assert_equal(uint32_t(768), out[0].k_cols);
        t.assert_equal(uint32_t(3), out[1].kv_head_begin);
        t.assert_equal(uint32_t(18), out[1].q_head_begin);
        t.assert_equal(uint32_t(6), out[1].q_head_count);
        t.assert_equal(uint32_t(256), out[1].v_cols);
    });
    t.test("zero_head_device_is_allowed_but_empty_total_is_not", [](testing & t) {
        std::vector<llama_kv_stream_shard> out;
        t.assert_true(llama_kv_stream_shard_map_make(qwen({4, 0}), out).status == llama_kv_stream_shard_status::success);
        t.assert_equal(uint32_t(0), out[1].k_cols);
        t.assert_true(llama_kv_stream_shard_map_make(qwen({0, 0}), out).status == llama_kv_stream_shard_status::heads_sum_mismatch);
    });
    t.test("invalid_geometry_fails_closed_and_leaves_output_unchanged", [](testing & t) {
        std::vector<llama_kv_stream_shard> out(7);
        using s = llama_kv_stream_shard_status;
        t.assert_true(llama_kv_stream_shard_map_make(qwen({}), out).status == s::no_devices);
        t.assert_true(llama_kv_stream_shard_map_make(qwen({2, 1}), out).status == s::heads_sum_mismatch);
        t.assert_true(llama_kv_stream_shard_map_make(qwen({3, 2}), out).status == s::heads_sum_mismatch);
        auto p = qwen({2, 2}); p.n_head = 25;
        t.assert_true(llama_kv_stream_shard_map_make(p, out).status == s::bad_geometry);
        p = qwen({2, 2}); p.head_dim_k = 80; p.block_k = 64;
        auto r = llama_kv_stream_shard_map_make(p, out);
        t.assert_true(r.status == s::block_misaligned);
        t.assert_equal(uint32_t(0), r.device);
        p = qwen({2, 2}); p.n_head_kv = 0;
        t.assert_true(llama_kv_stream_shard_map_make(p, out).status == s::bad_geometry);
        t.assert_equal(size_t(7), out.size());
    });
    t.test("candidates_enumerate_all_ordered_splits", [](testing & t) {
        t.assert_equal(size_t(5), llama_kv_stream_shard_candidates(4, 2).size());
        t.assert_equal(size_t(15), llama_kv_stream_shard_candidates(4, 3).size());
        auto c = llama_kv_stream_shard_candidates(4, 2);
        t.assert_true(std::vector<uint32_t>({0, 4}) == c.front());
        t.assert_true(std::vector<uint32_t>({4, 0}) == c.back());
        for (auto & heads : c) {
            std::vector<llama_kv_stream_shard> out;
            t.assert_true(llama_kv_stream_shard_map_make(qwen(heads), out).status == llama_kv_stream_shard_status::success);
        }
    });
    return t.summary();
}
