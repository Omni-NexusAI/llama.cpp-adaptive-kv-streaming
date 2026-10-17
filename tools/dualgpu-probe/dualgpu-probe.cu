// Dual-GPU hardware probe for the KV streaming work: device identity, per-card pinned H2D/D2H bandwidth, peer access.
// It is inert unless the caller proves a written hardware release: --released-by-user AND DUALGPU_PROBE_HARDWARE_RELEASED=1.
// No CUDA call is made before that check (no static initializers touch CUDA).
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(e_), __FILE__, __LINE__, cudaGetErrorString(e_)); std::exit(2); } } while (0)

static void identity() {
    int n = 0;
    CK(cudaGetDeviceCount(&n));
    for (int d = 0; d < n; ++d) {
        cudaDeviceProp p;
        CK(cudaGetDeviceProperties(&p, d));
        char bus[32];
        CK(cudaDeviceGetPCIBusId(bus, sizeof(bus), d));
        CK(cudaSetDevice(d));
        size_t free_b = 0, total_b = 0;
        CK(cudaMemGetInfo(&free_b, &total_b));
        char uuid[40];
        for (int i = 0; i < 16; ++i) std::snprintf(uuid + 2 * i, 3, "%02x", (unsigned char) p.uuid.bytes[i]);
        std::printf("{\"device\":%d,\"name\":\"%s\",\"pci\":\"%s\",\"uuid\":\"%s\",\"cc\":\"%d.%d\",\"total_mib\":%zu,\"free_mib\":%zu,"
                    "\"async_engines\":%d,\"host_map\":%d,\"wddm_tcc\":%d}\n",
                    d, p.name, bus, uuid, p.major, p.minor, total_b >> 20, free_b >> 20, p.asyncEngineCount, p.canMapHostMemory, p.tccDriver);
    }
    for (int a = 0; a < n; ++a) for (int b = 0; b < n; ++b) if (a != b) {
        int can = 0;
        CK(cudaDeviceCanAccessPeer(&can, a, b));
        std::printf("{\"peer\":[%d,%d],\"can_access\":%d}\n", a, b, can);
    }
}

struct bw_result { double h2d_gbs = 0, d2h_gbs = 0; };

static bw_result bandwidth_one(int dev, size_t mib, int iters) {
    CK(cudaSetDevice(dev));
    const size_t bytes = mib << 20;
    void * host = nullptr, * device = nullptr;
    CK(cudaHostAlloc(&host, bytes, cudaHostAllocPortable));
    CK(cudaMalloc(&device, bytes));
    std::memset(host, 0x5a, bytes);
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    bw_result r;
    for (int dir = 0; dir < 2; ++dir) {
        CK(cudaMemcpyAsync(device, host, bytes, cudaMemcpyHostToDevice, s)); // warm-up
        CK(cudaStreamSynchronize(s));
        CK(cudaEventRecord(a, s));
        for (int i = 0; i < iters; ++i) {
            if (dir == 0) CK(cudaMemcpyAsync(device, host, bytes, cudaMemcpyHostToDevice, s));
            else          CK(cudaMemcpyAsync(host, device, bytes, cudaMemcpyDeviceToHost, s));
        }
        CK(cudaEventRecord(b, s));
        CK(cudaEventSynchronize(b));
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, a, b));
        const double gbs = double(bytes) * iters / 1e9 / (ms / 1e3);
        (dir == 0 ? r.h2d_gbs : r.d2h_gbs) = gbs;
    }
    CK(cudaEventDestroy(a));
    CK(cudaEventDestroy(b));
    CK(cudaStreamDestroy(s));
    CK(cudaFree(device));
    CK(cudaFreeHost(host));
    return r;
}

static void bandwidth(const std::vector<int> & devs, size_t mib, int iters, bool concurrent) {
    std::vector<bw_result> out(devs.size());
    if (concurrent) {
        std::vector<std::thread> threads;
        for (size_t i = 0; i < devs.size(); ++i) threads.emplace_back([&, i] { out[i] = bandwidth_one(devs[i], mib, iters); });
        for (auto & t : threads) t.join();
    } else {
        for (size_t i = 0; i < devs.size(); ++i) out[i] = bandwidth_one(devs[i], mib, iters);
    }
    for (size_t i = 0; i < devs.size(); ++i) {
        std::printf("{\"bandwidth\":{\"device\":%d,\"concurrent\":%s,\"mib\":%zu,\"iters\":%d,\"h2d_gbs\":%.2f,\"d2h_gbs\":%.2f}}\n",
                    devs[i], concurrent ? "true" : "false", mib, iters, out[i].h2d_gbs, out[i].d2h_gbs);
    }
}

int main(int argc, char ** argv) {
    bool flag = false, identity_mode = false, bw_mode = false, concurrent = false;
    size_t mib = 256;
    int iters = 20;
    std::vector<int> devs;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--released-by-user") flag = true;
        else if (a == "--identity") identity_mode = true;
        else if (a == "--bandwidth") bw_mode = true;
        else if (a == "--concurrent") concurrent = true;
        else if (a == "--mib" && i + 1 < argc) mib = (size_t) std::atoi(argv[++i]);
        else if (a == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (a == "--devices" && i + 1 < argc) {
            std::string list = argv[++i];
            size_t pos = 0;
            while (pos <= list.size()) {
                size_t comma = list.find(',', pos);
                if (comma == std::string::npos) comma = list.size();
                if (comma > pos) devs.push_back(std::atoi(list.substr(pos, comma - pos).c_str()));
                pos = comma + 1;
            }
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", a.c_str());
            return 1;
        }
    }
    const char * env = std::getenv("DUALGPU_PROBE_HARDWARE_RELEASED");
    if (!flag || !env || std::strcmp(env, "1") != 0) {
        std::fprintf(stderr, "dualgpu-probe is inert: it needs --released-by-user and DUALGPU_PROBE_HARDWARE_RELEASED=1. No CUDA call was made.\n");
        return 3;
    }
    if (!identity_mode && !bw_mode) {
        std::fprintf(stderr, "choose --identity and/or --bandwidth --devices 0,1 [--mib N --iters K --concurrent]\n");
        return 1;
    }
    if (identity_mode) identity();
    if (bw_mode) {
        if (devs.empty() || mib == 0 || iters <= 0) {
            std::fprintf(stderr, "--bandwidth needs --devices and positive --mib/--iters\n");
            return 1;
        }
        bandwidth(devs, mib, iters, concurrent);
    }
    return 0;
}
