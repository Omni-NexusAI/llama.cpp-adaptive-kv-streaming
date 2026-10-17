# Build record: CUDA 12.4.1 SM86 compile-only tree

Source: `db8fd5d898b0d19be74a4269c126c23d2f62c38d` (tag `pin/upstream-v2-db8fd5d`), unmodified.
Toolchain: MSVC 19.29.30159 (VS2019 BuildTools 14.29.30133), Windows SDK 10.0.19041.0, CUDA 12.4.131 (toolkit 12.4.1), CMake 3.31.10, Ninja.
Result: configure and build of `llama-server` succeeded (416/416 steps). The binary was NOT executed (not even --version).

Configure (key variables):
- `-G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86-real -DGGML_NATIVE=OFF`
- `-DGGML_CUDA_FA_QUANTS="f16-f16;bf16-bf16;q8_0-q8_0;q8_0-q4_0;q4_0-q4_0" -DGGML_CUDA_GRAPHS=OFF -DGGML_CUDA_NCCL=OFF`
- `-DBUILD_SHARED_LIBS=OFF -DGGML_BACKEND_DL=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_UI=OFF`
- CMake reported `Using CMAKE_CUDA_ARCHITECTURES=86-real`; "Check for working CUDA compiler" was skipped, so no GPU probe ran at configure time.

Static checks (cuobjdump, no GPU): 164 embedded `sm_86` cubins, no PTX, as expected for `86-real`.
Notes: configure tried to download UI assets from GitHub (HTTP error, harmless with the UI off). No ccache. Tests are compiled only in the CPU tree; do not run tests from this tree.
Binary size 195 MB. Tree: `Z:\AI Runtimes\AdaptiveKV\build\sm86-v12.4-compile-only`.
