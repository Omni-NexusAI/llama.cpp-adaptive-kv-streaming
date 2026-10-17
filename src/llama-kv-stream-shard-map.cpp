#include "llama-kv-stream-shard-map.h"

#include <numeric>

using status = llama_kv_stream_shard_status;

llama_kv_stream_shard_result llama_kv_stream_shard_map_make(
        const llama_kv_stream_shard_params & params, std::vector<llama_kv_stream_shard> & output) {
    if (params.kv_heads.empty()) return {status::no_devices, UINT32_MAX};
    if (params.n_head == 0 || params.n_head_kv == 0 || params.n_head % params.n_head_kv != 0 ||
            params.head_dim_k == 0 || params.head_dim_v == 0 || params.block_k == 0 || params.block_v == 0) {
        return {status::bad_geometry, UINT32_MAX};
    }
    const uint32_t gqa = params.n_head / params.n_head_kv;
    uint64_t sum = 0;
    for (auto h : params.kv_heads) sum += h;
    if (sum != params.n_head_kv) return {status::heads_sum_mismatch, UINT32_MAX};
    std::vector<llama_kv_stream_shard> next(params.kv_heads.size());
    uint32_t begin = 0;
    for (uint32_t d = 0; d < next.size(); ++d) {
        const uint32_t heads = params.kv_heads[d];
        auto & s = next[d];
        s.kv_head_begin = begin;
        s.kv_head_count = heads;
        s.q_head_begin = begin * gqa;
        s.q_head_count = heads * gqa; // whole GQA groups by construction
        const uint64_t k_cols = uint64_t(heads) * params.head_dim_k, v_cols = uint64_t(heads) * params.head_dim_v;
        if (k_cols > UINT32_MAX || v_cols > UINT32_MAX) return {status::bad_geometry, d};
        if (k_cols % params.block_k != 0 || v_cols % params.block_v != 0) return {status::block_misaligned, d};
        s.k_cols = uint32_t(k_cols);
        s.v_cols = uint32_t(v_cols);
        begin += heads;
    }
    output = std::move(next);
    return {};
}

static void compose(uint32_t left, size_t slots, std::vector<uint32_t> & current,
        std::vector<std::vector<uint32_t>> & output) {
    if (slots == 1) {
        current.push_back(left);
        output.push_back(current);
        current.pop_back();
        return;
    }
    for (uint32_t h = 0; h <= left; ++h) {
        current.push_back(h);
        compose(left - h, slots - 1, current, output);
        current.pop_back();
    }
}

std::vector<std::vector<uint32_t>> llama_kv_stream_shard_candidates(uint32_t n_head_kv, uint32_t n_devices) {
    std::vector<std::vector<uint32_t>> output;
    if (n_head_kv == 0 || n_devices == 0) return output;
    std::vector<uint32_t> current;
    compose(n_head_kv, n_devices, current, output);
    return output;
}

std::vector<uint32_t> llama_kv_stream_stock_kv_heads(
        uint32_t n_head_kv, uint32_t head_dim, const std::vector<float> & ts, uint32_t rotation) {
    const size_t n = ts.size();
    std::vector<uint32_t> heads(n, 0);
    if (n == 0 || head_dim == 0) return heads;
    const int64_t ne = int64_t(n_head_kv) * head_dim, g = head_dim;
    std::vector<float> scan(n);
    for (size_t j = 0; j < n; ++j) {
        scan[j] = ts[(j + rotation) % n] + (j > 0 ? scan[j - 1] : 0.0f);
    }
    int64_t low = 0;
    size_t j = 0;
    for (; j + 1 < n; ++j) {
        int64_t high = scan.back() == 0.0f ? ne * int64_t(j + 1) / int64_t(n) : int64_t(ne * scan[j] / scan.back());
        if (high % g != 0) high -= high % g;
        heads[(j + rotation) % n] = uint32_t((high - low) / g);
        low = high;
    }
    heads[(j + rotation) % n] = uint32_t((ne - low) / g);
    return heads;
}
