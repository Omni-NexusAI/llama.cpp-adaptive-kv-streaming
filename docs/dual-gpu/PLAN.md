# Adaptive KV Streaming V2: true dual-GPU tensor split (plan)

Status: planning gate released 2026-10-17 for CPU-only preparation. GPU testing is NOT released.
Owner: Omni-NexusAI fork. Not for upstream submission.

## Pins

- Base: `db8fd5d898b0d19be74a4269c126c23d2f62c38d` (V2 head observed 2026-10-17 19:25Z, tag `pin/upstream-v2-db8fd5d`).
  It carries the generic KV-pair work (KV_QUANTIZATION_ROADMAP.md): Q8_0/Q8_0 and MTP admission, DLL export fixes.
- Earlier plan pin `1c4469df33c8a1bf7dbbc81cd2d23fcf54122ef8` (tag `pin/upstream-v2-1c4469d`) is kept for reference.
- Upstream llama.cpp base of V2: `10a60cf303566e10d6a7a2774c17d2085503d87b`. Cherry-pick candidate: ggml-org/llama.cpp#30453 (meta subgraph realloc). Check #30103 against the private tile kernels.
- PR #2 of the V2 repo (multi-GPU, Conrad-N) is based on V1. Reuse ideas only, not code.

## Findings that shape the design

1. V2 is single-device in the context guard, the KV cache constructor, the config validator, and the arena coordinator
   (`llama_context_memory::supported()` refuses the meta backend and more than one CUDA backend; streaming returns false without it).
2. Stock tensor split shards the KV cache by KV head (`cache_(k|v)_l*`, axis 0, granularity one head = 256).
   With `-ts 35,65` and 4 KV heads, attention layers alternate 1/3 and 2/2 heads per device. V2 needs one shape per device.
3. Under tensor split, FA needs Q head-split and K/V head-split (or both mirrored), mask mirrored. SET_ROWS needs equal split states.
   The meta backend has no API for externally owned shard storage. Windows has no NCCL (generic all-reduce).
4. PCIe (Windows-reported, to be confirmed by a bandwidth test): RTX 3070 Gen4 x16 (bus 8), RTX 3060 Gen3 x4 (bus 4).
5. Build hazards: `CMAKE_CUDA_ARCHITECTURES=native` probes the GPU at configure time. CUDA test targets touch CUDA before they skip.

## Decisions

- D1 Pin above. No mainline rebase now. Cherry-pick #30453 with a test.
- D2 Shard map `(layer, device) -> KV-head range` and a `llama_kv_stream_group` of per-device models.
  begin/complete/abort/truncate/restore/reset are all-or-nothing. Same code serves layer split and tensor split.
- D3 Head map uniform per device across all attention layers incl. NextN. Gated override of attention split states (off = stock byte-identical).
  Chosen from measured H2D bandwidth and free VRAM, not from `-ts`. Candidates (3070:3,3060:1), (2,2), (1,3).
- D4 Fail closed: non-integral heads, shard not a multiple of the GQA group (6 Q heads) or quant block, per-layer width change,
  parallel>1, `--fit`, DFlash2, vision (initially), unqualified KV pairs.
- D5 Pinned shard-major host planes per device. Logical-row adapter for state I/O.
- D6 New meta API: derived meta buffer type from per-device stream buffer types, bind shard storage. Unique buffer-type names per device.
- D7 Stage 1: bypass the arena coordinator in group mode (private pools, stock compute buffers, eager, `GGML_CUDA_GRAPHS=OFF`).
  Stage 2: N-device coordinator with per-device phase transitions and meta subgraph invalidation.
- D8 Per-device streams and events. Any shard or all-reduce failure aborts all shards and fails the group until reconstructed.
- D9 KV pairs: Q8_0/Q4_0 and Q8_0/Q8_0 both run the ladder. Report quality, speed, capacity per pair.
- D10 Three build trees. CPU-only tree runs tests. CUDA SM86 tree is compile-only and never executed during the GPU restriction.

## Reference model

Hardware tests use one model: Swift 1.5 Qwen3.8 27B UD-Q4_K_S (14.30 GiB, only Swift quant installed). It carries the MTP head.
APEX I-Mini (1.32 GiB lighter) and UD-Q4_K_M (1.03 GiB heavier) are covered by the budget model, not by hardware runs.

## Milestones

M0 pin, toolchain record, hook audit, CPU tree, CUDA compile-only tree, test census.
M1 shard map, head-map override, validation. M2 group coordinator, shard planes, state adapter (mocks).
M3 meta shard-binding API and #30453 (CPU mock-device run). M4 CUDA wiring behind a flag (bypass, eager).
M5 layer-split shakedown (not success). M6 tensor-split text baseline. M7 N-device arena. M8 MTP. M9 vision. M10 CUDA graphs.

## Acceptance (GPU parts pending hardware release)

Gates: Windows SM86 compile; device identity; forced streaming proven by per-device counters; long retrieval and multi-turn;
prefix reuse, cancel, exhaustion, repeated-run stability.
Ladder: 8K, 32K, 64K, 96K, 128K, 160K, 192K, 256K in tiers with stop rules (Tier 3 only if Tier 2 passes).
Tolerances (locked after CPU-reference calibration): KV planes 0 differing bytes; MMA span paths bit-exact; vector/tile <= 1e-8 relative;
final logits bitwise or max abs diff <= 1e-4 with identical top-5; last-layer hidden <= 1e-6 relative; greedy 256 IDs identical on 3 prompts.
Pool invariance: logits identical across at least 3 pool sizes. Quality per KV pair: KL and top-1 vs F16 KV at 8K-32K, needle recall 64K-256K.
Hardware floor: 512 MiB free per card until a measured desktop margin replaces it.

## Hardware coordination (after release)

Written release per card. Preflight: no other CUDA process, ports free, free VRAM at or above floor. Order: identity, one card, bandwidth,
synthetic two-card tests, layer split, tensor split. Loopback test port, one server, time-boxed, stop only own process, log failures.

## Rollback

Delete `C:\AI Runtimes\AdaptiveKV\<tag>` and the experimental launcher; reset to the pin tag. Production runtimes, router, Hermes card, and keys are never touched.
