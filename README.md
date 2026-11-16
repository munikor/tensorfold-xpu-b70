# TensorFold XPU on Intel Arc Pro B70

A build recipe and measured results for TensorFold's XPU backend — a Zig
inference engine that drives an Intel Battlemage card through Level Zero, with
no PyTorch, no SYCL and no Python at serve time.

- [Recipe](TENSORFOLD-XPU.md) — toolchain, the nGEN header closure, the
  `Product` constructor patch, and the exact serve command.
- [BetterBench chart](tensorfold-xpu-175-betterbench.html) — decode, concurrency
  and prefill sweeps measured on host 175.

## Headline numbers (one B70, Qwen3.8-27B GPTQ-INT4-g128 + DFlash2)

| Axis | Result |
|---|---|
| Decode, weighted median across six categories | 125.4 tok/s |
| Decode, concurrency 1 / 2 / 4 / 8 | 108 / 173 / 209 / **234** tok/s aggregate |
| Prefill, 1.5K / 5.9K / 11.8K prompts | 1,621 / 1,812 / 1,789 tok/s |
| Peak observed single response | 161.7 tok/s |
| Card power under load | 220.6 W of a 230 W cap |
| GPU clock under load | 2,500 MHz (= `max_freq`; `rp0_freq` 2,800) |

Measured 2026-11-16, BetterBench corpus v1.0, 20 passes per category, greedy,
cold prefix cache. Self-report, C1 evidence.

## Scope and limits

- The XPU backend is **single-device**: it enumerates every card but drives one.
- It serves `qwen3_5` (Qwen3.8-27B), `nemotron_h`, `lfm3` and `gemma4` only. It
  does **not** serve `qwen4_exp` (Qwen3.8-Flash-Next).
- Context tops out at **16,384 tokens** on one card: 262,144 is the engine's
  ceiling, but weights plus KV plus the 2 GiB margin exceed 30.3 GiB at that
  depth, so admission refuses it.
- 62 of 160 BetterBench runs stopped at `max_tokens` (39%), so on a thinking
  model those measured the reasoning phase rather than a complete answer. The
  weighted median is the honest figure; the per-category mean is inflated.
