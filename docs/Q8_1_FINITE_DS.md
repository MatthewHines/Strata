# q8_1 activation blocks: keeping the scale and sum finite (#606, follow-up)

Status: follow-up to the fixes in `19de7f4` / `f6601bf`, which introduced
`include/strata/kernels/q8_1_finite.hpp` and routed the main q8_1 activation
emit sites through it. This note covers the three sites that were missed and
the change that clamps them the same way.

## The invariant

A q8_1 activation block stores 32 `int8` quants plus an fp16 `half2` pair:
the block scale `d = amax / 127.0f` and the sum-of-absolutes term `sum` the
dequant path consumes alongside it. Both halves must be finite fp16 values.
The dequant and dot kernels do not test for finiteness; they assume the
producer upheld the invariant, because every producer shares the helper
`q8_1_ds()` from `q8_1_finite.hpp`, which clamps `d` into the fp16-finite
range and clamps `sum` to a non-negative finite value (it is a mean of `|x|`,
so it is never legitimately negative).

## What the three missed sites do differently

| site | kernel | path |
|---|---|---|
| `src/kernels/cuda/fused_gr.cu`, `gr_q8_tail` | fused gate-reduce tail | the fused (QFUSE) expert path |
| `src/kernels/cuda/verify_kernels.cu`, `gdn_q8_1_store` | batched GDN verify store | speculative-window store on the fused path |
| `src/kernels/cuda/iq_kernels.cu`, `s26_swiglu_q8_1_kernel` | S26 swiglu quantize | the S26 expert layout |

Each wrote the pair with a raw `make_half2(d, sum)`. When a block's `amax`
overflows — or the kernel reads a non-finite input — `d` is `inf`/NaN, and
`__float2half` turns the pair into NaN (or the pathological `(0, 0)`, which
silently reads the block as all-`-127 * 0` bias). There is no per-use guard
downstream: a NaN scale propagates through the fp16 dot, through the expert
output, and through every residual sum that follows it in the layer stack.

The end of that chain is the sampler. When the damage reaches the logits, the
greedy path's all-non-finite fallback selects vocabulary id 0 — on
Qwen-family vocabularies the `!` token — and a row that is *mostly* damaged
picks arguments from noise. That is the visible face of the degeneration
reports in #606: runs of `!`, and content collapse onto whatever tokens the
perturbed logits favor.

## Why this is a correctness fix, not a filter

The clamp runs at the producer, at the numeric layer, and is entirely
content-blind: it changes what bits a damaged *activation block* can carry,
never what text the model can say. It converts a NaN-poisoning block into a
finite, neutral one — the same bits the CPU expert pool and the already-fixed
GPU sites produce for the same inputs. This matters for the class of bug in
#606: text-level repetition detectors treat symptoms; the engine must not
invent damage in the first place.

## Diagnostic criteria

Evidence that a deployment is hitting one of these sites:

1. Degeneration output begins mid-conversation and is cured by a full engine
   restart while the same prompt re-reads clean — damage lives in carried
   engine state (checkpoints, caches), not in the weights or the prompt.
2. Once begun it affects *every* session, including brand-new ones: sessions
   fork from a shared root checkpoint and inherit the damaged state until it
   is dropped.
3. The visible token is often vocabulary id 0 (`!` on Qwen tokenizers — the
   sampler's all-non-finite fallback), or a small set of literal tokens with
   no structure (content-id collapse, e.g. runs of `.`/`1`/`3`).
4. On this branch family the fused/S26 expert paths are environment-gated;
   reproductions concentrate where those paths are enabled (the performance
   branch), while default production boots run the clamped path.

(1)–(3) together point at carried-state damage; a one-token id-0 face points
specifically at a non-finite activation scale reaching the sampler.

## Related findings

The same investigation localized a second, independent violation of the
engine's determinism contract — expert-output bits depending on which tier
(GPU-resident vs CPU pool) an expert was assigned, because the two paths
quantize the activation scale at different precision — and a third one
through the speculative drafting policy. Those are discussed in
`docs/DEGENERATION_RCA.md` alongside the reproduction and verification
protocol, because the remedy for all three is the same invariant:

*anything that chooses **how** to compute is part of the output function
unless the numerics are provably shape-invariant — otherwise it must be
frozen, keyed, or shared.*
