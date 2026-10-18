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
