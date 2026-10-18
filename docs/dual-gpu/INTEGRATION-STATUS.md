# Dual-GPU integration status (2026-10-17)

Evidence classes: SOURCE = read only; BUILD = compiled; CPU = CPU or mock test passed; GPU = executed on a GPU (none yet).
Base `db8fd5d898b0d19be74a4269c126c23d2f62c38d`. Reference model: Swift 1.5 Qwen3.8 27B UD-Q4_K_S. KV pairs: Q8_0/Q4_0 and Q8_0/Q8_0.

## Done without a GPU

| Piece | Branch | Evidence |
|---|---|---|
| Plan, CPU baseline census, SM86 build record | `dualgpu/00-plan` | SOURCE, BUILD (CUDA 12.4.1 + MSVC 19.29; binaries never executed) |
| `llama-kv-stream-shard-map`: whole-KV-head ranges, GQA and quant-block checks, candidates, stock rotation mirror, head split helper, request validation with 15 named errors | `dualgpu/10-shard-map` | CPU: `test-kv-stream-shard-map` |
| Gated per-device KV head map in `llama_meta_device_get_split_state` (`--kv-stream-heads`; empty = stock) | `dualgpu/10-shard-map` | BUILD only (needs a loaded model to run) |
| `llama_kv_stream_group`: all-or-nothing begin/complete/abort/truncate/restore/reset, one frontier, reconstruct | `dualgpu/20-group` | CPU: `test-kv-stream-group` (10 mock cases) |
| Meta backend: #30453, name fix, `..._with_bufts`, `..._tensor_shard`, `..._buffer_from_buffers`, `..._tensor_alloc_shards` | `dualgpu/30-meta` | CPU: `test-backend-meta-shards`, `test-backend-meta` (mock devices) |
| Model-to-member adapter, group wiring in context, KV cache and hybrid memory (per-device models, meta shard binding, coordinator bypass, group begin/complete/abort, reset/seq_rm/state restore through the group), `--kv-stream-pool-mib N,N` | `dualgpu/40-wiring` | BUILD (CPU tree and CUDA SM86 compile-only tree). Not executed |
| `dualgpu-probe` (identity, bandwidth, peer); inert without `--released-by-user` and `DUALGPU_PROBE_HARDWARE_RELEASED=1` | `dualgpu/40-wiring` | BUILD only |

CPU census after the wiring: 61 tests selected, same 7 not passing and 2 skipped as the baseline (see `records/CPU-BASELINE-2026-10-17.md`). No regression, no new pass either: the wiring has no CPU test.

## Group mode contract (fail closed)

On only when `--kv-stream-heads` and a per-device `--kv-stream-pool-mib` list are given with `--split-mode tensor`. Requires: qwen35, flash attention on, KV offload, `--fit off`, one sequence, no MTP, no rollback, no shared arena, no vision, no DFlash, head size 256, 256-token pages, whole GQA groups, every device with at least one KV head.
Compute buffers stay stock; each shard model keeps a private pool and attention workspace. Eager execution (build with `GGML_CUDA_GRAPHS=OFF`).
Any member failure aborts every shard and fails the group until `reconstruct`. Reset, seq_rm and state restore recover a failed group by reconstructing to the requested frontier.

## Not exercised (needs a GPU)

- Execution-buffer hooks on each meta sub-backend, pinned shard planes, DMA ordering, per-device policy behavior.
- Whether the meta backend's per-device subgraphs reach the shard owner for flash attention and KV writes as designed.
- The head-map override on a real model (attention tensors must split identically across q/k/v/o/cache).
- Pool sizing against real free VRAM; streaming proof counters; all numerical tolerances.
- Not implemented: MTP, vision, devices with zero KV heads, per-device phase arena, CUDA graphs.

Next hardware steps need no model: G1 identity and G3 bandwidth (`HARDWARE-GATES.md`).
