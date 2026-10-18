#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

// KV-head shard map for multi-device KV streaming. Pure metadata: no backend, no allocation of device memory.
// Rule: every device owns one contiguous range of whole KV heads, the same range on every attention layer.

struct llama_kv_stream_shard_params {
    uint32_t n_head = 0;      // Q heads per layer
    uint32_t n_head_kv = 0;   // KV heads per layer
    uint32_t head_dim_k = 0;
    uint32_t head_dim_v = 0;
    uint32_t block_k = 1;     // elements per quant block of the K type (ggml_blck_size)
    uint32_t block_v = 1;     // elements per quant block of the V type
    std::vector<uint32_t> kv_heads; // KV heads per device, device order
};

struct llama_kv_stream_shard {
    uint32_t kv_head_begin = 0, kv_head_count = 0;
    uint32_t q_head_begin = 0, q_head_count = 0;
    uint32_t k_cols = 0, v_cols = 0; // row width in elements of this device's K and V plane
};

enum class llama_kv_stream_shard_status {
    success, no_devices, bad_geometry, heads_sum_mismatch, gqa_not_integral, block_misaligned,
};

struct llama_kv_stream_shard_result {
    llama_kv_stream_shard_status status = llama_kv_stream_shard_status::success;
    uint32_t device = UINT32_MAX; // offending device, or UINT32_MAX when not device-specific
};

// A device with zero heads is allowed (it holds no KV and no attention slice). All other rules fail closed.
llama_kv_stream_shard_result llama_kv_stream_shard_map_make(
        const llama_kv_stream_shard_params & params, std::vector<llama_kv_stream_shard> & output);

// Every ordered split of n_head_kv over n_devices (zeros allowed), for candidate search.
std::vector<std::vector<uint32_t>> llama_kv_stream_shard_candidates(uint32_t n_head_kv, uint32_t n_devices);

// Split `ne_s` elements (one segment, one repeat) over devices in proportion to KV head counts, device d getting
// ne_s*heads[d]/n_head_kv. Fails (output untouched) when heads do not sum to n_head_kv or ne_s is not a multiple of n_head_kv.
bool llama_kv_stream_head_split(int64_t ne_s, const std::vector<uint32_t> & heads, uint32_t n_head_kv, int64_t * output);

// Everything the group mode must reject before any device memory is touched. Pure data, so it can be tested without a GPU.
struct llama_kv_stream_group_request {
    bool arch_qwen35 = false;
    bool tensor_split = false;
    bool flash_attn = false;
    bool kv_offload = false;
    bool fit = false;
    bool vision = false;
    bool dflash = false;
    bool embeddings = false;
    bool state_on_device = false;
    bool recurrent_rollback_unsupported = false;
    uint32_t n_seq_max = 1;
    uint32_t n_devices = 0;
    uint32_t page_tokens = 256;
    llama_kv_stream_shard_params shard;
};

enum class llama_kv_stream_group_status {
    ok, wrong_arch, not_tensor_split, single_device, no_flash_attn, no_kv_offload, fit_enabled, vision_unsupported, dflash_unsupported,
    embeddings_unsupported, parallel_unsupported, head_list_size, head_dim_unsupported, page_geometry, shard_map_invalid,
};

llama_kv_stream_group_status llama_kv_stream_group_validate(const llama_kv_stream_group_request & request);
const char * llama_kv_stream_group_status_text(llama_kv_stream_group_status status);

// Mirror of the stock meta split rounding for one cache_k/cache_v tensor (axis 0, granularity one KV head),
// used only to show what stock gives and to test the override. `ts` are device fractions; `rotation` is il_eff % n_devices.
std::vector<uint32_t> llama_kv_stream_stock_kv_heads(
        uint32_t n_head_kv, uint32_t head_dim, const std::vector<float> & ts, uint32_t rotation);
