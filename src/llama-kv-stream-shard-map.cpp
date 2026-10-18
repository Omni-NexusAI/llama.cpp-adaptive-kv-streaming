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

bool llama_kv_stream_head_split(int64_t ne_s, const std::vector<uint32_t> & heads, uint32_t n_head_kv, int64_t * output) {
    if (!output || heads.empty() || n_head_kv == 0 || ne_s <= 0 || ne_s % n_head_kv != 0) return false;
    uint64_t sum = 0;
    for (auto h : heads) sum += h;
    if (sum != n_head_kv) return false;
    const int64_t per_head = ne_s / n_head_kv;
    for (size_t d = 0; d < heads.size(); ++d) output[d] = per_head * heads[d];
    return true;
}

llama_kv_stream_group_status llama_kv_stream_group_validate(const llama_kv_stream_group_request & r) {
    using g = llama_kv_stream_group_status;
    if (!r.arch_qwen35) return g::wrong_arch;
    if (!r.tensor_split) return g::not_tensor_split;
    if (r.n_devices < 2) return g::single_device;
    if (!r.flash_attn) return g::no_flash_attn;
    if (!r.kv_offload) return g::no_kv_offload;
    if (r.fit) return g::fit_enabled;
    if (r.vision) return g::vision_unsupported;
    if (r.dflash) return g::dflash_unsupported;
    if (r.embeddings) return g::embeddings_unsupported;
    if (r.n_seq_max != 1 || r.state_on_device) return g::parallel_unsupported;
    if (r.shard.kv_heads.size() != r.n_devices) return g::head_list_size;
    if (r.shard.head_dim_k != 256 || r.shard.head_dim_v != 256) return g::head_dim_unsupported;
    if (r.page_tokens != 256) return g::page_geometry;
    std::vector<llama_kv_stream_shard> map;
    if (llama_kv_stream_shard_map_make(r.shard, map).status != llama_kv_stream_shard_status::success) return g::shard_map_invalid;
    return g::ok;
}

const char * llama_kv_stream_group_status_text(llama_kv_stream_group_status s) {
    using g = llama_kv_stream_group_status;
    switch (s) {
        case g::ok: return "ok";
        case g::wrong_arch: return "multi-GPU KV streaming supports only the qwen35 architecture";
        case g::not_tensor_split: return "multi-GPU KV streaming needs --split-mode tensor";
        case g::single_device: return "--kv-stream-heads needs at least two devices";
        case g::no_flash_attn: return "multi-GPU KV streaming needs flash attention on";
        case g::no_kv_offload: return "multi-GPU KV streaming needs KV offload";
        case g::fit_enabled: return "multi-GPU KV streaming needs --fit off (fit is not implemented for tensor split)";
        case g::vision_unsupported: return "multi-GPU KV streaming does not support multimodal projectors yet";
        case g::dflash_unsupported: return "multi-GPU KV streaming does not support DFlash drafters";
        case g::embeddings_unsupported: return "multi-GPU KV streaming does not support embedding mode";
        case g::parallel_unsupported: return "multi-GPU KV streaming needs one sequence and no on-device state transfer";
        case g::head_list_size: return "--kv-stream-heads must list one entry per device";
        case g::head_dim_unsupported: return "multi-GPU KV streaming supports head size 256 only";
        case g::page_geometry: return "multi-GPU KV streaming supports 256-token pages only";
        case g::shard_map_invalid: return "--kv-stream-heads does not give whole GQA groups that sum to the KV head count and align to the quant block";
    }
    return "unknown";
}
