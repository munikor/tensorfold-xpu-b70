# Qwen3.8-27B on one B70 via TensorFold XPU (Level Zero, no Python at run time)

> Route record — measured 2026-11-16 on host 175, 2× Arc Pro B70 (single card
> used). Self-report, C1 evidence. This is a **new engine route**, not a patch
> stack: TensorFold's XPU backend talks Level Zero directly, with no PyTorch,
> no SYCL and no Python at serve time.

## What it is

[TensorFold](https://github.com/ashhart/TensorFold) is a Zig inference engine.
Its **unofficial XPU backend** (`docs/xpu/`) drives a Battlemage card through
Level Zero and serves `qwen3_5` behind the `tensorfold-native` OpenAI API with
the DFlash2 tree drafter.

The backend serves **only** these families on XPU:

| family | formats |
|---|---|
| `qwen3_5` (Qwen3.8-27B) | `mlx-q4g64`, `gptq-int4-g128` |
| `nemotron_h` | `mlx-q4g64` |
| `lfm3` | `mlx-q4g64` |
| `gemma4` | `mlx-q4g64` |

It does **not** serve `qwen4_exp` (Qwen3.8-Flash-Next) — use the inferzero or
radiance routes for that model.

- Engine: TensorFold `1.2.9`, `zig build xpu-native` (see *Build* below).
- Weights (pinned): `SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16`
  rev `e28c5f952bdd5d814297a07d85a064a87af26a3f` — GPTQ 4-bit, group 128,
  symmetric, 18.24 GiB, MTP tensors present but unused by this route.
- Drafter: `z-lab/Qwen3.8-27B-DFlash2` safetensors directory
  (`DFlash2DraftModel`, `target_layer_ids [5,19,33,47]`).
- Hardware: 1× Intel Arc Pro B70 (30.3 GiB usable). The XPU backend is
  **single-device** — it enumerates every device but drives one.

## Build

The XPU target is not vendored with its nGEN dependency. oneDNN tags
**v3.5-v3.7** still carry `src/gpu/intel/jit/ngen/`; v3.8+ removed it. Fetch the
full header closure from `uxlfoundation/oneDNN` at `v3.7` into one `<ngen>` dir:

```text
ngen.hpp ngen_core.hpp ngen_elf.hpp ngen_utils.hpp ngen_interface.hpp
ngen_config.hpp ngen_gen8.hpp ngen_gen12.hpp ngen_registers.hpp
ngen_register_decl.{hpp,cpp} ngen_register_allocator.{hpp,cpp}
ngen_auto_swsb.hpp ngen_compiler_fix.hpp ngen_debuginfo.hpp ngen_decoder.hpp
ngen_level_zero.hpp ngen_opencl.hpp ngen_pseudo.hpp COPYRIGHT
npack/{neo_packager,elf_structs,neo_structs,hash}.hpp      # all four
common/{bfloat16,float16,bit_cast}.hpp
oneapi/dnnl/{dnnl.h,dnnl_common.h,dnnl_common_types.h,dnnl_types.h}
oneapi/dnnl/dnnl_config.h  oneapi/dnnl/dnnl_version.h      # hand-written
```

`dnnl_config.h` and `dnnl_version.h` are cmake-generated and absent from git.
Do **not** mechanically rewrite the `.in` template: `#cmakedefine
DNNL_CPU_RUNTIME` expands to a literal `${DNNL_CPU_RUNTIME}` token and gcc
rejects `{` in a `#if` expression. Hand-write a minimal header (`DNNL_API`
empty, runtimes `DNNL_RUNTIME_NONE`, version 3/7/0).

v3.7's `ngen_core.hpp` declares `struct Product { ProductFamily family; int
stepping; }` with no constructor, but TensorFold calls
`Product(ProductFamily::GenericXe2, 0, PlatformType::Discrete)`. Add a shim and
move `enum class PlatformType` **above** `struct Product` (v3.7 declares it
after):

```cpp
enum class PlatformType {Unknown, Integrated, Discrete};
struct Product {
    ProductFamily family;
    int stepping;
    PlatformType platformType = PlatformType::Unknown;
    Product() = default;
    Product(ProductFamily f, int s = 0, PlatformType pt = PlatformType::Unknown)
        : family(f), stepping(s), platformType(pt) {}
};
```

Safe because `Product` is only compared through `.family`/`.stepping`.

Toolchain, all user-local:

- Zig **0.17.0 exactly** (the repo's `.zig-version`). Never wrap it as `c++`:
  Zig resolves `c++` from `PATH`, so a wrapper named `c++` that calls `zig c++`
  recurses into itself and hangs.
- A real `c++` (g++ is fine).
- `ocloc` from the matching compute-runtime release (26.40.1 here), with its
  `libocloc.so` on `LD_LIBRARY_PATH`.
- Intel Level Zero 26.40 + **IGC 2.42.6** staged in a private dir. B70
  (`0xE223`) needs that IGC; older 2.28.x aborts in
  `command_stream_processor.cpp` with `preemptionSurfaceSize == 0`.

```bash
cd <repo> && PATH=<zig-dir>:$PATH \
  LD_LIBRARY_PATH=<ocloc-lib>:<igc-lib> \
  zig build xpu-native -Doptimize=ReleaseFast \
    -Dngen-include=<ngen> -Docloc=<path-to-ocloc>
```

Artifact: `zig-out/xpu/bin/tensorfold-native` (~46 MB, statically linked
against libc only). Verify before serving:

```bash
tensorfold-native capabilities --json
# -> "chip":"Intel Xe2 (Level Zero)", "backends":["xpu"]
```

## Serve

```bash
tensorfold-native pull SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16

tensorfold-native serve \
  SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16 \
  --host 0.0.0.0 --port 8240 \
  --context 16384 --parallel 8 \
  --kv-cache fp8 \
  --drafter <dir-with-model.safetensors> \
  --no-thinking
```

`--drafter` takes a **directory containing `model.safetensors`**, not a GGUF.
The z-lab DFlash2 GGUF will not load.

## Measured (one B70, host 175, 2026-11-16)

BetterBench v1.0 corpus, 20 passes per category, greedy, cold prefix cache:
[charted report](assets/tensorfold-xpu-175-betterbench.html).

| Axis | Result |
|---|---|
| Load time | 5.7 s (warm kernel cache) |
| Streams / context | 7 streams × 16,384 tok (`--parallel 8`, one slot held) |
| Decode, weighted median across six categories | **125.4 tok/s** |
| Decode, concurrency 1 | 108.0 tok/s aggregate, 140.7 per-request, TTFT p50 200 ms |
| Decode, concurrency 2 / 4 / 8 | 172.6 / 209.3 / **234.3 tok/s** aggregate |
| Prefill, 1,516-token prompt | 1,621 tok/s, TTFT 929 ms |
| Prefill, 5,894-token prompt | 1,812 tok/s, TTFT 3,251 ms |
| Prefill, 11,802-token prompt | 1,789 tok/s, TTFT 6,589 ms |
| Peak observed single response | 161.7 tok/s (119 tokens, 102/110 drafts accepted) |
| Card power under load | 220.6 W of a 230 W cap |
| GPU clock under load | 2500 MHz actual = `max_freq` (`rp0_freq` 2800) |

Resident memory: 20.3 GiB host RSS; weights 18.24 GiB; prompt cache 2.0 GiB
budget, ~0.19 GiB per kept state.

Two caveats from the harness: **62 of 160 runs stopped at `max_tokens`** (39%),
so on a thinking model those measured the reasoning phase rather than a
complete answer — the per-category `mean` column is inflated accordingly, and
the weighted median is the honest figure. Every run streamed several tokens per
update, so per-token inter-token latency is not reported.

## Context limits on one card

`MAX_CONTEXT` is 262,144 — the engine's ceiling. It does **not** fit beside the
weights on one B70. Admission (`zig/src/native/xpu_admit.zig`) requires

```
free - margin(2 GiB) - prompt_cache >= slot_bytes
```

where `slot_bytes = 48*(2*48*128*128*4 + 2*3*10240*2) + 16*2*kvBytes()` and
`kvBytes = 262144 * 4 * (fp8 ? 260 : 512)`. At 262K that is 8.41 GiB (fp8) or
16.29 GiB (bf16) per stream against ~10 GiB free — both refused. 16,384 tok
fits comfortably; 32,768 tok fits at fp8 with `--parallel 4`.

## Gotchas

- A **baked** community checkpoint (extra `model-bake-int4.safetensors`, INT8
  embedding side file) loads but is not detected as `gptq-int4-g128`. Use the
  canonical `SergiioB` artifact.
- `tensorfold-native info` prints `format: full precision` for a quantized
  checkpoint; the benchmark receipt correctly reports
  `quantization: gptq-4bit-g128`. Trust the receipt.
- The benchmark receipt labels `runtime.backend` as `cuda` regardless of the
  build — the real backend is whatever `capabilities --json` reports.
- Thinking is **on** by default from the chat template. It roughly halves
  decode throughput because reasoning tokens count against `max_tokens`.
- `libtfgrammar.so` is missing from a source build, so structured output and
  required tool calls return HTTP 400. Build it with
  `zig build grammar -Dxgrammar=<dir>` against
  `git clone --branch v0.2.8 --recurse-submodules https://github.com/mlc-ai/xgrammar`.

## Claim boundary

Single-card, single-host, self-report at C1. No prefill numbers, no
concurrency sweep, no exact-answer battery yet. Do not compare these decode
figures against the EXL3 or GPTQ-INT4+vLLM routes without a matched run — the
cache states, drafters, and thinking settings differ.
