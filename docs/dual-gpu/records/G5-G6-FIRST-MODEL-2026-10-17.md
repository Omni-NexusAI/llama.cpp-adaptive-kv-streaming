# First dual-GPU tensor-split streamed KV runs (GPU evidence)

Model: Swift 1.5 Qwen3.8 27B UD-Q4_K_S. Base `db8fd5d` + `dualgpu/40-wiring`. Eager (CUDA graphs off), CUDA 12.4.1 SM86, MSVC 19.29.
Common flags: `--split-mode tensor -ts 26,74 -ngl 99 -fa on --cache-type-k q8_0 --cache-type-v q4_0 --fit off --parallel 1 --batch-size 256 --ubatch-size 128 --threads 8 --kv-stream-heads 1,3 --cache-ram 0`.
Device order is CUDA0 = RTX 3070 (1 KV head), CUDA1 = RTX 3060 (3 KV heads). `-ts 26,74` is chosen from today's free VRAM (3070 had about 5 GiB free), not tuned.
Guard: a sampler killed the server below 256 MiB free on either card; it never fired. Minimum free during runs: 3060 968-1566 MiB, 3070 1515-1659 MiB.

## What it took to get the first token (bugs found by running)
1. `create_memory` passed stream parameters only for the single-device pool flags, so group mode silently built a stock cache. Fixed.
2. The scheduler had no backend for the meta-level FLASH_ATTN_EXT node. The meta device now admits attention and KV-write nodes whose cache lives in execution-buffer shards; each device's owner still admits its own shard at compute time.
3. The V write needs a paired source that the graph builder only added when the cache buffer itself was an execution buffer. It now also does so for execution-sharded meta storage.
4. Known open bug: with the RAM prompt cache on, the server dies silently when the second request starts (state save path). All runs below use `--cache-ram 0`.

## Equality: streamed group vs non-streamed control (same binary, same head map, same split, same KV types)
| Context | Pools (3070 / 3060) | Prompts | Result |
|---|---|---|---|
| 8192, pools larger than shards (all resident) | 192 / 384 MiB | 5 prompts incl. 589-token prefill, 28-48 greedy tokens each, top-5 logprobs | tokens identical, max logprob difference 0, top-5 order identical |
| 32768, pools smaller than shards (streaming forced) | 96 / 256 MiB (shards 208 / 624 MiB) | needles at 7,001 / 14,012 / 21,023 tokens, depths 0.50 / 0.25 / 0.75 | all found; tokens identical, max logprob difference 0, top-5 order identical |

Streaming proof from the server log at 21K tokens: 3070 shard 83 active pages, 57 resident + 33 ring slots, 41 MiB over 208 H2D calls per evaluation; 3060 shard 83 active pages, 50 resident + 40 ring slots, 157 MiB over 394 H2D calls.

## Speed (small pools, eager, not tuned; indicative only)
| Context used | Streamed decode t/s | Control decode t/s | Streamed prefill t/s | Control prefill t/s |
|---|---|---|---|---|
| 7K | 14.1 | 15.4 | 372 | 392 |
| 14K | 12.2 | 15.2 | 372 | 388 |
| 21K | 7.1 | 14.7 | 350 | 381 |
Short prompts at 8K: 13.5-14.3 streamed vs 16.2 control.

Files: harness and raw JSON in `Z:\AI Runtimes\AdaptiveKV\logs\eq*.json`.
