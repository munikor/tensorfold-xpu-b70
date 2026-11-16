# TensorFold XPU on Intel Arc Pro B70

Build recipe and measured results for TensorFold's XPU backend — a Zig inference
engine driving an Intel Battlemage card through Level Zero, with no PyTorch, no
SYCL and no Python at serve time.

- [Recipe](TENSORFOLD-XPU.md) — toolchain, the nGEN header closure, the
  `Product` constructor patch, and the exact serve command.
- [BetterBench chart](assets/swift-tf-200k.html) — decode, concurrency and
  prefill sweeps at 200,000-token context.

## Headline numbers (one B70, Swift 1.5 27B GPTQ-INT4-g128 + DFlash2, 200K ctx)

| Axis | Result |
|---|---|
| Decode, weighted median across eight categories | 125.2 tok/s |
| Decode, best category (math / json / code) | 159.2 / 157.4 / 148.8 tok/s |
| Prefill, 1.5K / 5.9K / 11.8K prompts | 1,675 / 1,838 / 1,810 tok/s |
| Prefill, 23.5K / 47K prompts | 1,719 / 1,573 tok/s |
| TTFT p50 | 168 ms |
| Peak observed single response | 159 tok/s |
| Card power under load | 220.6 W of a 230 W cap |
| GPU clock under load | 2,500 MHz (= `max_freq`; `rp0_freq` 2,800) |

Measured 2026-11-16, BetterBench corpus v1.0, 20 passes per category, greedy,
cold prefix cache. Self-report, C1 evidence.

## The concurrency ceiling

At 200K context **only one stream fits**, so aggregate throughput is flat:

| concurrency | aggregate tok/s | TTFT p50 |
|---|--:|--:|
| 1 | 112.9 | 165 ms |
| 2 | 110.3 | 1,886 ms |
| 4 | 110.9 | 6,743 ms |
| 8 | 110.7 | 14,255 ms |

Requests queue; latency grows, throughput does not. `--parallel N` is accepted
but admission silently caps it — the server logs `1 streams of 200000 tokens`.
For concurrency, drop context: 65K fits 2 slots, 32K fits 4.

## Scope and limits

- The XPU backend is **single-device**: it enumerates every card but drives one.
- It serves `qwen3_5` (Qwen3.8-27B and Swift 1.5 27B), `nemotron_h`, `lfm3` and
  `gemma4` only. It does **not** serve `qwen4_exp` (Qwen3.8-Flash-Next).
- `--kv-cache fp8` is **ignored** on this backend — set `XPU_KV_INT8=1` instead.
  Without it, 200K is refused outright (12.49 GiB/slot vs 6.49 GiB).
- 59 of 160 BetterBench runs stopped at `max_tokens` (37%), so on a thinking
  model those measured the reasoning phase rather than a complete answer. The
  weighted median is the honest figure.
