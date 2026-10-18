# G2 single-card CUDA tests (GPU evidence)

Build: CUDA 12.4.1, MSVC 19.29, `86-real`, `GGML_CUDA_GRAPHS=OFF`, base `db8fd5d` plus the dual-GPU branch. Each test ran with `CUDA_VISIBLE_DEVICES` set to one card.
Both cards: same results.

| Test | RTX 3060 | RTX 3070 |
|---|---|---|
| `test-cuda-fattn-native --backend CUDA0` | 70 cases passed | 70 cases passed |
| `test-kv-stream-model --cuda-native-admission` | 5 tests, 341 assertions, 0 failures | same |
| `test-memory-executor-cuda --cuda` | 10 of 10 fail: all graph capture cases (this build has graphs off) | not run |
| `test-memory-executor-cuda --cuda --no-graphs` | 9 of 10 pass; `partial_submission_failure_keeps_pins_and_closes_admission` fails 2 assertions | same |

The executor failure is in code this branch does not touch. It is recorded as a baseline GPU finding on Windows/MSVC/CUDA 12.4, not investigated yet.
Not yet built or run: `test-cuda-compiled-features` (needed `CUDA_STANDARD 17`), `test-kv-stream-copy` (cudart static/shared clash), vector-spans, context, capture. CMake fixes in `tests/CMakeLists.txt`.

## Second batch (RTX 3060, `GGML_CUDA_PDL=0`, after the CMake fixes)

| Test | Result |
|---|---|
| `test-cuda-compiled-features` | passed (no GPU needed) |
| `test-kv-stream-copy --cuda --queue-only` | 1460 assertions, 0 failures. Pinned-host DMA, event ordering, teardown, canaries work on Windows |
| `test-kv-stream-vector-spans --cuda` | 2340 assertions, 25 failures in 5 cases (below) |
| `test-kv-stream-capture --cuda` | 85 assertions, 8 failures: CUDA-graph capture cases, expected with graphs off, not investigated |

vector-spans passes the cases that matter for this work: compiled cache pairs execute stock-order spanned attention (387), wide MMA dispatch (291), quantized vector stock equivalence (230), TG3/TG4 F16 MMA spans (99), Q8/Q4 tile conversion matches stock (147), Qwen ratio-six TG3/TG4 (25), Qwen 24/4 TG2 aligned and wrapped spans exact (48), TG1 fixed spans (14, 6).
Failing: `float32_storage_uses_bounded_stock_equivalent_attention` (18) and `float32_ring_waves_...` (1) (this build does not compile F32 pairs; not used), `stock_selected_plan_matches_workspace_and_launch` (1), `invalid_span_plan_and_workspace_preserve_output` (1),
`tg1_tg2_partition_boundaries_match_contiguous_and_stock` (4 of 95; reported max absolute error 2.8e-9 to 3.7e-9, under the fork's documented 1e-8 guard for regional TG1 reductions).
Cause of the last three not determined; the fork qualified CUDA 13 on an RTX 5070 Ti, this is CUDA 12.4 on SM86 Windows.
