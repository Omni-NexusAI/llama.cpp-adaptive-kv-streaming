# Swift 1.5 Qwen3.8 27B UD-Q4_K_S: 200K text + image + MTP on the streamed tensor split (2026-10-17/18)

Commit under test: `4f075622` (branch `dualgpu/40-wiring`), CUDA 12.4.1, MSVC 19.29, `86-real`, graphs off. Hardware: RTX 3070 8 GiB (display, Gen4 x16) and RTX 3060 12 GiB (Gen3 x4). Model: Swift UD-Q4_K_S, `-ts 24,76`, `--kv-stream-heads 3,1`, K Q8_0 / V Q4_0, one slot, `-b 256 -ub 128`, `--cache-ram 0`. Floor for this profile: 256 MiB free per card (an agreed exception to 512 MiB).

## What changed in the engine since the KV pair study
1. **Compute-buffer fix.** The Meta compute buffer grew by about 4 KiB per context token (153 MiB at 32K, 537 at 128K, 1,037 at 256K). Cause: the CUDA buffer type adds an F16 copy of the whole K/V to the flash-attention node's allocation size for quantized caches, and group mode never presents an execution owner, so the full estimate was always used. The shard owners run attention on their own workspaces, so that scratch never exists. The meta buffer type now reports only the node's output size for execution-sharded FA. Evidence: F16/F16 KV gave 35.5 MiB at 64K, Q8_0/Q4_0 gave 281 MiB, and after the fix Q8_0/Q4_0 gives 35.8 MiB at 64K and 256K alike.
2. **MTP in group mode.** Recurrent rollback (`n_rs_seq`) is allowed; the draft context keeps one plain tensor-split KV layer and ignores the per-device pools. `--spec-draft-kv-window N` keeps draft KV for the most recent N tokens (draft context sized N plus headroom). It changes draft acceptance only.
3. **Vision in group mode.** Uses the stock projector path (BF16 projector on CPU), not the single-GPU vision arena.

## Correctness
| Check | Result |
|---|---|
| MTP, 32K, pools 96/64 (forced streaming) vs stock tensor-split MTP, 5 short prompts | tokens identical; draft acceptance counts identical (e.g. 33/42) |
| MTP, 32K, needles at 7K/14K/21K vs the same control | tokens identical; max logprob difference 3.3e-6 |
| Vision, 32K, 2 questions vs non-streamed control | answers and all logprobs identical (difference 0) |
| Vision, 128K context, image first then 100,000 tokens of text, forced streaming | both answers correct (red square on blue; green rectangle) |
| Vision + MTP, same 128K test | both answers correct; drafts accepted during answers (9/9, 9/12) |
| Text ladder, 256K context (pools 640/256), needles at 60K/120K/180K | all found; decode 7.6 / 3.4 / 2.2 t/s; prefill 325 / 215 / 159 t/s; min free 1,376 / 578 MiB |
| Multi-turn prefix reuse through the router, 20K-token prefix | follow-up reprocessed 31 of 20,076 tokens; repeat reprocessed 4; answers correct |

## Speed with MTP (window 8K, 128K context, pools 640/256)
| Depth (tokens) | No MTP | MTP |
|---|---|---|
| 30K | 13.8 | 26.2 |
| 60K | 8.3 | 18.5 |
| 90K | 4.7 | 13.3 |
Draft acceptance about 0.65 on repetitive filler text at depth (0.79 to 0.89 at 32K with a full-context draft). Not measured on real prose or code.

## Sweet spot (full profile: vision projector loaded, MTP, window 8K, pools 3070 640 / 3060 384 MiB)
| Context | Needles (tokens) | Decode t/s | Prefill t/s | Min free 3060 / 3070 (MiB) |
|---|---|---|---|---|
| 200K | 37.5K / 75K / 112.5K | 19.9 / 16.4 / 12.4 | 315 / 279 / 225 | 560 / 594 |
| 128K | 30K / 60K / 90K | 26.0 / 25.2 / 15.6 | 355 / 312 / 259 | 558 / 452 |
**Matched-depth check (same 30K / 60K / 90K prompts, same flags, one run each):** 128K decodes 26.0 / 25.2 / 15.6 t/s, 200K decodes 21.2 / 21.9 / 14.8 t/s (5 to 18 percent slower; prefill 3 to 4 percent slower; draft acceptance equal at 0.65 to 0.67; min free 558 / 584 MiB at 200K). The cause of the small gap is not identified and single runs carry noise, so treat it as a range.

VRAM use is the same at 128K and 200K; context size now costs pinned host RAM only (about 3.3 GiB at 128K, 5.0 at 200K, 6.5 at 256K of KV history). Decode at a given depth does not depend on the context setting, so 200K is the larger usable window at no VRAM cost. The 3060 pool matters: enlarging it from 256 to 384 MiB took decode at 128K/60K from 18.5 to 25.2 t/s. A 3060 pool of 1,280 MiB, and one of 640 MiB with the full profile, failed to allocate at load: nvidia-smi free overstates what cudaMalloc can get by a few hundred MiB near the limit, so keep several hundred MiB above the allocation being made.

## Staged router check (loopback port 8085, not the production port; 200K and 128K presets)
`/v1/models` listed the preset; the model loaded on the first request (about 25 s with cached weights); short text, an image question and a 40K-token needle all answered correctly through `/v1/chat/completions`; 24 to 27 t/s at shallow depth; min free 558 / 651 MiB (3060 / 3070). The 128K preset passed the same checks (text, image, 40K needle; min free 558 / 563 MiB). Production router not touched.

## Limits and open items
- The 3070 drives the display: its free memory dipped as low as 282 MiB at 200K when the desktop took 3 to 4 GiB. The guard killed nothing at 256 MiB, but there is little margin. Close heavy GPU apps before loading.
- A device with zero KV heads (for example (4,0), which would remove the slow 3060 link from the KV path) is rejected by the group code: "does not support a device without KV heads yet". Likely the largest remaining speed lever at depth.
- The windowed draft is untested for acceptance on real text, and the draft's own KV is not streamed. Streaming the draft layer would allow a full-context draft.
- The `--cache-ram` prompt cache (host RAM) stays off: it crashed the group server on a second request earlier and is not fixed. Slot-level prefix reuse works (see above).
- Not tested: real photographs, more than one slot, repeated-run memory stability, MTP with more than 3 drafts, a Hermes client session, laptop access.
- CPU test census after these changes: 61 selected, the same 7 fail and 2 skip as the pre-change baseline; all new tests pass.
- Line endings: earlier commits on this stack rewrote LF files as CRLF (Python text-mode edits on Windows). Commit `a64329dc` restores the base style; net diff against base is 1,967 insertions and 60 deletions across 43 files. The intermediate stacked PR diffs (#2 to #4) still show that churn.

## Rollback
Nothing here changes the production router, runtimes, launchers, Hermes config or secrets. Delete `launchers\adaptivekv\` and `Z:\AI Runtimes\AdaptiveKV\runtime\adaptivekv-4f075622\` to remove the staged profile. Revert the fork branch to `ce59ff77` to undo the code changes.
