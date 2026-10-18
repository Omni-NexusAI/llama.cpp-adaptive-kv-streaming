# KV pair study: Q8_0/Q4_0 versus Q8_0/Q8_0 (GPU evidence, Swift UD-Q4_K_S, dual-GPU streamed)

Setup as in `G5-G6-FIRST-MODEL-2026-10-17.md`: head map (3070, 3060) = (3, 1), `-ts 24,76`, eager, CUDA 12.4.1 SM86. Pools 640 / 256 MiB at 128K.

## Correctness (both pairs)
At 32K, a non-streamed control, a 96/64 MiB pool and a 320/160 MiB pool give identical tokens and a maximum logprob difference of exactly 0 for needle prompts of 7,001 / 14,012 / 21,023 tokens (both pairs). Streaming and pool size change speed only.

## Quality against an F16 KV baseline (llama-perplexity, 8 chunks x 4096 tokens of repo documentation text, 16,384 scored tokens)
| Pair | PPL | Mean KLD | Same top token | RMS delta-p |
|---|---|---|---|---|
| F16/F16 (baseline) | 13.170 +/- 0.291 | - | - | - |
| Q8_0/Q8_0 | 13.187 (+0.13%) | 0.00253 +/- 0.00007 | 96.90% | 1.09% |
| Q8_0/Q4_0 | 13.202 (+0.24%) | 0.00448 +/- 0.00006 | 95.70% | 1.42% |
Q4_0 V costs about 1.8x the KL of Q8_0 V; both keep perplexity within a quarter of a percent at 4K. This measures cache-quantization error at 4K context on technical English; it does not measure long-context accumulation (see needle retrieval) or other text types.

## Speed at 128K context (indicative: small pools, eager, not tuned)
| Context used (tokens) | Q8_0/Q4_0 decode / prefill t/s | Q8_0/Q8_0 decode / prefill t/s |
|---|---|---|
| 30K | 13.8 / 372 | 13.5 / 374 |
| 60K | 8.3 / 333 | 6.0 / 309 |
| 90K | 4.7 / 272 | 3.6 / 253 |
Needles at 30K / 60K / 90K tokens found by both pairs. The gap grows with depth because Q8_0/Q8_0 moves 31% more bytes per token through PCIe.

## Capacity
| | Q8_0/Q4_0 | Q8_0/Q8_0 |
|---|---|---|
| KV bytes per token (16 layers) | 26,624 | 34,816 |
| Pinned host KV at 128K | 3.25 GiB | 4.25 GiB |
| Pinned host KV at 256K | 6.50 GiB | 8.50 GiB |
| VRAM at 128K (pools fixed, 3070 640 MiB / 3060 256 MiB), min free MiB 3060 / 3070 | 880 / 1168 | 880 / 1166 |
Host RAM on this machine is 47.9 GB; with the model mapped and 4.25 GiB pinned, free RAM was about 6.4 GB, so 256K at Q8_0/Q8_0 is the pair most at risk.

## Reading
Q8_0/Q4_0: smaller, faster at depth, slightly lower quality. Q8_0/Q8_0: better quality, 31% more PCIe traffic, about 24% slower decode at 60K-90K with these pools. Pick by whether the extra 0.002 KL matters more than decode speed at depth; neither pair failed retrieval.

## Context ladder beyond 128K (Q8_0/Q4_0, heads 3,1, -ts 24,76, ubatch 128, eager, one slot)
| ctx | pools 3070/3060 MiB | needles (tokens) | decode t/s | prefill t/s | compute buffer / device | min free 3060 / 3070 MiB |
|---|---|---|---|---|---|---|
| 128K | 640/256 | 30K, 60K, 90K: all found | 13.8 / 8.3 / 4.7 | 372 / 333 / 272 | 537 | 880 / 1168 |
| 160K | 640/256 | 39K, 78K, 117K: all found | 9.0 / 5.4 / 3.4 | 349 / 267 / 199 | 665 | 747 / 344 |
| 200K | 320/256 | 49K, 98K, 146K: all found | 7.6 / 4.0 / 2.6 | 303 / 217 / 163 | 825 | 594 / 283 |
| 256K | 640/256 | not run: server died allocating a 1,049 MiB compute buffer on the 3060 | - | - | 1037-1049 | 392 free at load (3060), below floor |

Floor status: the 512 MiB floor on the 3070 was breached at 160K (344) and 200K (283); the 3060 held it. The 3070 is the display card and its desktop use varied between 1.7 and 3.9 GiB during these runs (3.9 GiB at the start of the 200K load, 3070 had only 586 MiB free after load). The server's own growth during inference was roughly 380 MiB at 160K. These rungs are functional passes (retrieval correct), not floor-compliant retained profiles; a floor-compliant 160K+ profile needs the desktop's 3070 use to be low and a smaller 3070 pool or weight share. Not retained.

## Open finding: compute buffer grows with context
Meta compute buffer per device: 153 MiB (32K), 537 (128K), 665 (160K), 825 (200K), 1037 (256K) = about 25 MiB + 4 KiB per context token, independent of ubatch (ubatch 64 gives the same 1037 MiB). No non-view node in the reserve graph scales with n_ctx (diagnostic env KVS_DEBUG_BIG_NODES, local), so the growth comes from the scheduler or meta layer. At head map (1,3) the 128K buffer was only 57 MiB, so it depends on the head map. Cause not found; fixing it would free about 1 GiB on the 3060 at 256K and is the main capacity lever.
