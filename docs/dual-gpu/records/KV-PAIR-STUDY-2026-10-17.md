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
