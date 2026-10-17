#include "../ggml/src/ggml-backend-impl.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// Mock devices for the meta backend: host memory, distinct names, one custom ("stream") buffer type per device.
struct mock_device {
    std::string dev_name, default_buft_name, stream_buft_name;
    ggml_backend_buffer_type default_buft = {}, stream_buft = {};
    ggml_backend_device device = {};
    size_t stream_allocs = 0, default_allocs = 0;
    bool is_stream_buft(ggml_backend_buffer_type_t b) const { return b == &stream_buft; }
};

struct mock_buffer { mock_device * owner; uint8_t * data; };

static size_t g_frees = 0;
static void buf_free(ggml_backend_buffer_t b) { auto * c = (mock_buffer *) b->context; ++g_frees; delete[] c->data; delete c; }
static void * buf_base(ggml_backend_buffer_t b) { return ((mock_buffer *) b->context)->data; }
static ggml_status buf_init(ggml_backend_buffer_t, ggml_tensor *) { return GGML_STATUS_SUCCESS; }
static void buf_set(ggml_backend_buffer_t, ggml_tensor * t, const void * d, size_t off, size_t n) { std::memcpy((uint8_t *) t->data + off, d, n); }
static void buf_get(ggml_backend_buffer_t, const ggml_tensor * t, void * d, size_t off, size_t n) { std::memcpy(d, (const uint8_t *) t->data + off, n); }
static void buf_clear(ggml_backend_buffer_t b, uint8_t v) { std::memset(buf_base(b), v, b->size); }

static ggml_backend_buffer_t buft_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    auto * owner = (mock_device *) buft->context;
    (owner->is_stream_buft(buft) ? owner->stream_allocs : owner->default_allocs)++;
    auto * c = new mock_buffer{owner, new uint8_t[size ? size : 1]};
    const ggml_backend_buffer_i iface = {
        /* .free_buffer   = */ buf_free, /* .get_base      = */ buf_base, /* .init_tensor   = */ buf_init,
        /* .memset_tensor = */ nullptr, /* .set_tensor    = */ buf_set, /* .get_tensor    = */ buf_get,
        /* .set_tensor_2d = */ nullptr, /* .get_tensor_2d = */ nullptr, /* .cpy_tensor    = */ nullptr,
        /* .clear         = */ buf_clear, /* .reset         = */ nullptr,
    };
    return ggml_backend_buffer_init(buft, iface, c, size);
}
static const char * buft_name(ggml_backend_buffer_type_t b) {
    auto * o = (mock_device *) b->context;
    return o->is_stream_buft(b) ? o->stream_buft_name.c_str() : o->default_buft_name.c_str();
}
static size_t buft_align(ggml_backend_buffer_type_t) { return 16; }
static bool buft_host(ggml_backend_buffer_type_t) { return false; }
static const char * dev_name(ggml_backend_dev_t d) { return ((mock_device *) d->context)->dev_name.c_str(); }
static void dev_mem(ggml_backend_dev_t, size_t * f, size_t * t) { *f = 0; *t = 0; }
static enum ggml_backend_dev_type mock_type(ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_GPU; }
static void dev_props(ggml_backend_dev_t d, ggml_backend_dev_props * p) { p->name = dev_name(d); p->description = dev_name(d); p->type = mock_type(d); }
static ggml_backend_buffer_type_t dev_buft(ggml_backend_dev_t d) { return &((mock_device *) d->context)->default_buft; }
static bool dev_supports_op(ggml_backend_dev_t, const ggml_tensor *) { return true; }
static bool dev_supports_buft(ggml_backend_dev_t d, ggml_backend_buffer_type_t b) {
    auto * o = (mock_device *) d->context;
    return b == &o->default_buft || b == &o->stream_buft;
}

static void make_device(mock_device & m, const char * name) {
    m.dev_name = name;
    m.default_buft_name = std::string("Default") + name;
    m.stream_buft_name = std::string("Stream") + name;
    for (auto * b : {&m.default_buft, &m.stream_buft}) {
        b->iface.get_name = buft_name; b->iface.alloc_buffer = buft_alloc;
        b->iface.get_alignment = buft_align; b->iface.is_host = buft_host;
        b->device = &m.device; b->context = &m;
    }
    m.device.iface = {
        /* .get_name = */ dev_name, /* .get_description = */ dev_name, /* .get_memory = */ dev_mem,
        /* .get_type = */ mock_type, /* .get_props = */ dev_props, /* .init_backend = */ nullptr,
        /* .get_buffer_type = */ dev_buft, /* .get_host_buffer_type = */ nullptr, /* .buffer_from_host_ptr = */ nullptr,
        /* .supports_op = */ dev_supports_op, /* .supports_buft = */ dev_supports_buft, /* .offload_op = */ nullptr,
        /* .event_new = */ nullptr, /* .event_free = */ nullptr, /* .event_synchronize = */ nullptr,
    };
    m.device.context = &m;
}

// KV cache tensor [cols, rows]: split on axis 0 into 768 + 256 columns (3 + 1 heads of 256).
static ggml_backend_meta_split_state kv_state(const ggml_tensor *, void *) {
    ggml_backend_meta_split_state s;
    std::memset(&s, 0, sizeof(s));
    s.axis = GGML_BACKEND_SPLIT_AXIS_0;
    s.ne[0] = 768; s.ne[1] = 256;
    s.nr[0] = 1;
    s.n_segments = 1;
    return s;
}

#define CHECK(cond) do { if (!(cond)) { std::fprintf(stderr, "FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond); return 1; } } while (0)

int main() {
    static mock_device a, b;
    make_device(a, "MockA");
    make_device(b, "MockB");
    ggml_backend_dev_t simple[] = {&a.device, &b.device};
    ggml_backend_dev_t meta = ggml_backend_meta_device(simple, 2, kv_state, nullptr);
    ggml_backend_buffer_type_t stock = ggml_backend_dev_buffer_type(meta);
    CHECK(stock && std::string(ggml_backend_buft_name(stock)) == "Meta(DefaultMockA,DefaultMockB)");

    // derived buffer type: names differ from stock, result is cached, bad inputs are rejected
    ggml_backend_buffer_type_t pair[] = {&a.stream_buft, &b.stream_buft};
    ggml_backend_buffer_type_t derived = ggml_backend_meta_buffer_type_with_bufts(meta, pair, 2);
    CHECK(derived && derived != stock);
    CHECK(std::string(ggml_backend_buft_name(derived)) == "Meta(StreamMockA,StreamMockB)");
    CHECK(ggml_backend_buft_is_meta(derived));
    CHECK(ggml_backend_buft_get_device(derived) == meta);
    CHECK(ggml_backend_meta_buffer_type_with_bufts(meta, pair, 2) == derived);
    ggml_backend_buffer_type_t swapped[] = {&b.stream_buft, &a.stream_buft};
    CHECK(ggml_backend_meta_buffer_type_with_bufts(meta, swapped, 2) == nullptr);
    CHECK(ggml_backend_meta_buffer_type_with_bufts(meta, pair, 1) == nullptr);
    CHECK(ggml_backend_meta_buffer_type_with_bufts(meta, nullptr, 2) == nullptr);
    CHECK(ggml_backend_meta_buffer_type_with_bufts(&a.device, pair, 2) == nullptr);
    ggml_backend_dev_t other_simple[] = {&b.device, &a.device};
    ggml_backend_dev_t meta_ba = ggml_backend_meta_device(other_simple, 2, kv_state, nullptr);
    CHECK(ggml_backend_meta_buffer_type_with_bufts(meta_ba, pair, 2) == nullptr);

    // allocate a KV-shaped tensor through the derived type: shards live in the per-device stream buffers
    constexpr int64_t cols = 1024, rows = 64;
    ggml_init_params params = {8 * ggml_tensor_overhead(), nullptr, true};
    ggml_context_ptr ctx(ggml_init(params));
    ggml_tensor * k = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, cols, rows);
    ggml_set_name(k, "cache_k_l3");
    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), derived));
    CHECK(buf);
    CHECK(a.stream_allocs >= 1 && b.stream_allocs >= 1);
    CHECK(a.default_allocs == 0 && b.default_allocs == 0);

    ggml_tensor * s0 = ggml_backend_meta_tensor_shard(k, 0);
    ggml_tensor * s1 = ggml_backend_meta_tensor_shard(k, 1);
    CHECK(s0 && s1);
    CHECK(s0->ne[0] == 768 && s0->ne[1] == rows);
    CHECK(s1->ne[0] == 256 && s1->ne[1] == rows);
    CHECK(s0->buffer->buft == &a.stream_buft && s1->buffer->buft == &b.stream_buft);
    CHECK(ggml_backend_meta_tensor_shard(k, 2) == nullptr);
    CHECK(ggml_backend_meta_tensor_shard(nullptr, 0) == nullptr);
    ggml_tensor * plain = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4);
    CHECK(ggml_backend_meta_tensor_shard(plain, 0) == nullptr);

    // set/get through the meta tensor reassembles rows from the shards; each shard holds its own columns
    std::vector<uint16_t> src(cols * rows), dst(cols * rows, 0);
    for (size_t i = 0; i < src.size(); ++i) src[i] = uint16_t(i * 2654435761u >> 7);
    ggml_backend_tensor_set(k, src.data(), 0, src.size() * sizeof(uint16_t));
    ggml_backend_tensor_get(k, dst.data(), 0, dst.size() * sizeof(uint16_t));
    CHECK(src == dst);
    for (int64_t r : std::vector<int64_t>{0, 17, rows - 1}) {
        const uint16_t * row0 = (const uint16_t *) ((const uint8_t *) s0->data + r * s0->nb[1]);
        const uint16_t * row1 = (const uint16_t *) ((const uint8_t *) s1->data + r * s1->nb[1]);
        CHECK(std::memcmp(row0, &src[r * cols], 768 * sizeof(uint16_t)) == 0);
        CHECK(std::memcmp(row1, &src[r * cols + 768], 256 * sizeof(uint16_t)) == 0);
    }
    // externally owned shard buffers: planes at explicit pointers, buffers survive the meta buffer
    {
        ggml_backend_buffer_ptr ext_a(ggml_backend_buft_alloc_buffer(&a.stream_buft, 1 << 20));
        ggml_backend_buffer_ptr ext_b(ggml_backend_buft_alloc_buffer(&b.stream_buft, 1 << 20));
        ggml_backend_buffer_t externals[] = {ext_a.get(), ext_b.get()};
        ggml_backend_buffer_t wrong[] = {ext_b.get(), ext_a.get()};
        CHECK(ggml_backend_meta_buffer_from_buffers(derived, wrong, 2, 4) == nullptr);
        CHECK(ggml_backend_meta_buffer_from_buffers(derived, externals, 1, 4) == nullptr);
        CHECK(ggml_backend_meta_buffer_from_buffers(stock, externals, 2, 4) == nullptr);
        const size_t frees_before = g_frees;
        {
            ggml_backend_buffer_ptr meta_buf(ggml_backend_meta_buffer_from_buffers(derived, externals, 2, 4));
            CHECK(meta_buf && ggml_backend_buffer_is_meta(meta_buf.get()));
            ggml_context_ptr ctx2(ggml_init(params));
            ggml_tensor * kk = ggml_new_tensor_2d(ctx2.get(), GGML_TYPE_F16, cols, rows);
            ggml_set_name(kk, "cache_k_l7");
            void * planes[] = {(uint8_t *) ggml_backend_buffer_get_base(ext_a.get()) + 4096,
                               (uint8_t *) ggml_backend_buffer_get_base(ext_b.get()) + 8192};
            void * bad[] = {planes[0], nullptr};
            CHECK(ggml_backend_meta_tensor_alloc_shards(meta_buf.get(), kk, bad, 2) == GGML_STATUS_FAILED);
            CHECK(ggml_backend_meta_tensor_alloc_shards(meta_buf.get(), kk, planes, 1) == GGML_STATUS_FAILED);
            CHECK(ggml_backend_meta_tensor_alloc_shards(meta_buf.get(), kk, planes, 2) == GGML_STATUS_SUCCESS);
            ggml_tensor * e0 = ggml_backend_meta_tensor_shard(kk, 0);
            ggml_tensor * e1 = ggml_backend_meta_tensor_shard(kk, 1);
            CHECK(e0 && e1 && e0->data == planes[0] && e1->data == planes[1]);
            CHECK(e0->buffer == ext_a.get() && e1->buffer == ext_b.get());
            CHECK(e0->ne[0] == 768 && e1->ne[0] == 256);
            ggml_backend_tensor_set(kk, src.data(), 0, src.size() * sizeof(uint16_t));
            std::vector<uint16_t> back(cols * rows, 0);
            ggml_backend_tensor_get(kk, back.data(), 0, back.size() * sizeof(uint16_t));
            CHECK(back == src);
            CHECK(std::memcmp((uint8_t *) planes[1] + 5 * e1->nb[1], &src[5 * cols + 768], 256 * sizeof(uint16_t)) == 0);
        }
        CHECK(g_frees == frees_before);
        ggml_backend_buffer_clear(ext_a.get(), 1);
    }
    std::printf("test-backend-meta-shards: all checks passed\n");
    return 0;
}
