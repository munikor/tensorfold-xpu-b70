# Swift 1.5 27B on one B70 via TensorFold XPU (Level Zero, no Python at run time)

Route record — measured 2026-11-16 on host 175, 1× Arc Pro B70. Self-report,
C1 evidence.

## What it is

[TensorFold](https://github.com/ashhart/TensorFold) is a Zig inference engine.
Its **unofficial XPU backend** (`docs/xpu/`) drives a Battlemage card through
Level Zero and serves `qwen3_5` behind the `tensorfold-native` OpenAI API with
the DFlash2 tree drafter.

Families served on XPU: `qwen3_5` (Qwen3.8-27B and Swift 1.5 27B) in
`mlx-q4g64` or `gptq-int4-g128`; `nemotron_h`, `lfm3`, `gemma4` in
`mlx-q4g64` only. It does **not** serve `qwen4_exp` (Qwen3.8-Flash-Next).

- Engine: TensorFold `1.2.9`, `zig build xpu-native`.
- Weights: `bjonor/Swift-1.5-Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16`
  rev `61411c73170317a5e199ae388c3b4042e7950d54` — GPTQ 4-bit, group 128,
  symmetric, 18.24 GiB.
- Drafter: `z-lab/Qwen3.8-27B-DFlash2` as a directory holding
  `model.safetensors` (`DFlash2DraftModel`, `target_layer_ids [5,19,33,47]`).
- Hardware: 1× Intel Arc Pro B70 (30.3 GiB usable, `max_freq` 2500 MHz,
  230 W cap). The XPU backend is single-device.

## Build

The XPU target is not vendored with its nGEN dependency. oneDNN tags
**v3.5-v3.7** still carry `src/gpu/intel/jit/ngen/`; v3.8+ removed it. Fetch the
full closure from `uxlfoundation/oneDNN` at `v3.7` into one `<ngen>` dir:

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
move `enum class PlatformType` **above** `struct Product`:

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

Toolchain, all user-local:

- Zig **0.17.0 exactly** (the repo's `.zig-version`). Never wrap it as `c++`:
  Zig resolves `c++` from `PATH`, so a wrapper named `c++` that calls `zig c++`
  recurses into itself and hangs.
- A real `c++` (g++ is fine).
- `ocloc` from the matching compute-runtime release (26.40.1), with its
  `libocloc.so` on `LD_LIBRARY_PATH`.
- Intel Level Zero 26.40 + **IGC 2.42.6** staged privately. B70 (`0xE223`)
  needs that IGC; older 2.28.x aborts in `command_stream_receiver.cpp` with
  `preemptionSurfaceSize == 0`.

```bash
cd <repo> && PATH=<zig-dir>:$PATH \
  LD_LIBRARY_PATH=<ocloc-lib>:<igc-lib> \
  zig build xpu-native -Doptimize=ReleaseFast \
    -Dngen-include=<ngen> -Docloc=<path-to-ocloc>
```

Artifact: `zig-out/xpu/bin/tensorfold-native` (~46 MB, linked against libc only).
Verify before serving:

```bash
tensorfold-native capabilities --json
# -> "chip":"Intel Xeon2 (Level Zero)", "backends":["xpu"]
```

## Serve at 200K context

**`--kv-cache fp8` is silently ignored on the XPU backend.** `zig/src/native/
xpu.zig` never passes `kv8` into the runner, so the cache stays bf16 and
`slotBytes` doubles. Set the env var instead:

```bash
tensorfold-native pull bjonor/Swift-1.5-Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16

XPU_KV_INT8=1 tensorfold-native serve \
  bjonor/Swift-1.5-Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16 \
  --host 0.0.0.0 --port 8240 \
  --context 200000 \
  --prompt-cache-gib 0.25 \
  --drafter <dir-with-model.safetensors> \
  --no-thinking
```

`--drafter` takes a **directory containing `model.safetensors`**, not a GGUF.

## Why only one stream

`slotBytes` allocates KV **per slot**, not shared:

```zig
// zig/src/xpu/slot_states.zig
return 48 * (2*48*128*128*4 + 2*3*10240*2) + 16 * 2 * kvBytes();
```

`kvBytes = ctx * 4 * (kv_i8 ? 260 : 512)`. At 200,000 tokens:

| KV | per slot |
|---|---|
| bf16 (512) | 12.49 GiB |
| fp8 (260, `XPU_KV_INT8=1`) | **6.49 GiB** |

Admission needs `free - 2 GiB margin - cache >= slotBytes`. After weights
(18.24 GiB) there is ~11 GiB free, so exactly **one** 6.49 GiB slot fits.
`--parallel 4` is accepted but capped — the server logs `1 streams of 200000
tokens`.

Context/slot trade on one B70 (fp8 KV):

| context | slots that fit |
|---|--:|
| 200K | 1 |
| 131K | 1 |
| 65K | 2 |
| 32K | 4 |

## Measured (one B70, 200K ctx, 2026-11-16)

BetterBench corpus v1.0, 20 passes per category, greedy, cold prefix cache:
[charted report](assets/swift-tf-200k.html).

| Category | decode t/s | TTFT p50 |
|---|--:|--:|
| math | 159.2 | 125 ms |
| json | 157.4 | 168 ms |
| code | 148.8 | 167 ms |
| file_edit | 137.7 | 168 ms |
| reasoning | 105.2 | 163 ms |
| chat | 105.0 | 155 ms |
| summarization | 104.8 | 197 ms |
| prose | 77.6 | 157 ms |

Weighted median **125.2 tok/s**, TTFT p50 168 ms, update p99 70.3 ms.

Concurrency (aggregate tok/s / TTFT p50): 1 → 112.9 / 165 ms · 2 → 110.3 /
1,886 ms · 4 → 110.9 / 6,743 ms · 8 → 110.7 / 14,255 ms. Flat throughput: one
slot.

Prefill: 1,516 tok → 1,675 t/s (913 ms) · 5,894 → 1,838 (3,216 ms) · 11,802 →
1,810 (6,529 ms) · 23,550 → 1,719 (13,724 ms) · 47,016 → 1,573 (29,906 ms).

Card under load: 220.6 W of 230 W; clock 2,500 MHz (= `max_freq`; `rp0` 2,800).

## Gotchas

- A **baked** checkpoint (extra `model-bake-int4.safetensors`, INT8 embedding
  side file) loads but is not detected as `gptq-int4-g128`. Use a canonical
  GPTQ export.
- `tensorfold-native info` prints `format: full precision` for a quantized
  checkpoint; the benchmark receipt correctly reports `gptq-4bit-g128`.
- The benchmark receipt labels `runtime.backend` as `cuda` regardless of build;
  trust `capabilities --json`.
- Thinking is **on** by default from the chat template and roughly halves
  decode throughput.
- `libtfgrammar.so` is missing from a source build, so structured output and
  required tool calls return HTTP 400. Build it with `zig build grammar
  -Dxgrammar=<dir>`.

## Claim boundary

Single-card, single-host, self-report at C1. No exact-answer battery. Do not
compare these figures against other B70 routes without a matched run — cache
states, drafters and thinking settings differ.
