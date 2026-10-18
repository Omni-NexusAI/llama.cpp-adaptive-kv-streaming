# Hardware gates (all PENDING; none may run without the user's written release per card)

Reference model: Swift 1.5 Qwen3.8 27B UD-Q4_K_S (14.30 GiB). KV pairs: Q8_0/Q4_0 and Q8_0/Q8_0. Floor: 512 MiB free per card until a measured desktop margin replaces it.
Windows-reported links (to confirm in G1): RTX 3070 Gen4 x16 (PCI bus 8), RTX 3060 Gen3 x4 (PCI bus 4). Stock CUDA order was 3070 = device 0, 3060 = device 1.

## Preflight for every window
- User confirms in writing which card(s) are released and for how long.
- No `llama-server` or other CUDA process on either card; ports 8080-8082 and the test port (loopback 8093) free.
- Free VRAM at or above floor plus the test need. Only one test server at a time. Stop only processes this run started. Confirm VRAM and port release after.
- Append a ledger row for every attempt, including failures.

## Gates
| Gate | What | Needs | Pass |
|---|---|---|---|
| G1 | Identity: UUID, PCI ID, device order, free memory, link state | `dualgpu-probe --identity` on one released card, then both | Order recorded; bus 8 and bus 4 mapped to CUDA indices |
| G2 | Single-card CUDA smoke from the compile-only tree | `test-cuda-compiled-features`, one card | No error |
| G3 | Pinned H2D and D2H bandwidth per card, then both at once | `dualgpu-probe --bandwidth --devices N` and `--concurrent` | GB/s table; decides the KV head map candidates |
| G4 | Synthetic two-card group tests, no model | wiring steps 1-3 | Group frontiers equal; fault injection recovers |
| G5 | Layer-split streaming shakedown on the reference model (not success) | wiring | Counters prove streaming |
| G6 | Tensor-split text baseline: 8K, 32K, 64K, 96K, 128K, then 160K-256K by tier | wiring | Equality tolerances, retrieval, lifecycle, 5 t/s decode threshold reported |
| G7 | KV pair study: quality (KL, top-1 vs F16 at 8K-32K; needle 64K-256K), speed, capacity | G6 | Table per pair |
| G8 | MTP: rejection, replay, cancellation | G6 | No divergence from non-MTP greedy |

G1 and G3 can run before any model integration. Tolerances are in PLAN.md.
