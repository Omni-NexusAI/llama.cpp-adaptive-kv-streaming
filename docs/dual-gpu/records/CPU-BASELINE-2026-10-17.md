# CPU-only baseline census (before any dual-GPU change)

Base `db8fd5d898b0d19be74a4269c126c23d2f62c38d`. Host: Windows 11, MSVC 19.29.30159 (VS2019 BuildTools), CMake 3.31.10, Ninja.
Tree: `GGML_CUDA=OFF`, static libs, backend DL off, `GGML_NATIVE=OFF` (AVX2/FMA/F16C on), tests on, server/tools/examples/OpenSSL/UI off.
No CUDA library is linked. No GPU was touched. These are CPU/mock results only, not runtime qualification.

Build: all `test-kv-stream-*` targets compile. Not built: `test-batch-alloc` (MSVC 19.29 C2131 at test-batch-alloc.cpp:916, upstream test),
`test-chat` (not yet analysed), and targets disabled on purpose (mtmd, app).

ctest `-R "kv-stream|memory|backend|executor|alloc|meta|split|state"`: 58 tests, 49 passed, 2 skipped (tile, tile-dispatch: need CUDA), 7 not passing:

| Test | Result | Likely cause (unconfirmed) |
|---|---|---|
| test-batch-alloc | not built | MSVC 19.29 constexpr rule |
| test-backend-memory | abort (0xc0000409) | GGML_ASSERT view_free_count == 1 in test_arena_commit_exception_safety |
| test-kv-stream-host | fail | pageable fallback / pinned allocation case needs CUDA pinned memory |
| test-kv-stream-mtp-lease | abort (0xc0000409) | not yet analysed |
| test-memory-recovery | abort (0xc0000409) | not yet analysed |
| test-memory-recurrent-spill | fail | "after packed recurrent submission" in 2 cases |
| test-save-load-state | segfault after many PASS rows | upstream synthetic-model test, not yet analysed |

These are the baseline. Any later change must not add to this list.
