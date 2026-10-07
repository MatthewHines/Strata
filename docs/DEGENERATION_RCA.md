# Degeneration on Qwen-family hybrid models: root-cause analysis and diagnostic criteria

Companion to `Q8_1_FINITE_DS.md`. This document records the RCA behind the
`#606` follow-up commits: how the carried-state degeneration was localized,
what the three distinct defects are, how each was confirmed with a
falsification test, and the criteria another deployment can use to identify
or rule out each one. Model paths, hostnames, and deployment-specific tuning
are omitted; everything here is reproducible on any affected engine.

## 1. Symptom

On a Qwen3-Next-class hybrid model (GDN/PLE recurrent state + sparse experts)
served with the conversation cache and speculative decoding enabled:

- output degrades mid-generation: runs of a single token (classically `!`),
  content collapse onto a few literal tokens, or structurally-shaped but
  meaningless text;
- a full engine restart cures it immediately — re-reading the *same* prompt
  on the restarted engine produces clean output;
- once onset occurs it affects **all** sessions, including brand-new ones,
  until the engine restarts or the conversation state is explicitly dropped
  (a `FLUSH`-style drop of the checkpoint chain cures identically).

Symptoms 2 and 3 are the strongest diagnostic levers. A restart cure means
the damage is **carried engine state**, not weights, not KV of one session,
not the prompt. Cross-session persistence narrows the carrier to state
shared by all sessions — the shared system-prompt root of the checkpoint
chain, from which every session forks.

## 2. The contract that was violated

The conversation cache memoizes the recurrent state: `state = f(prefix)`.
Forked checkpoints are trusted because the function is assumed pure. The
investigation found three mechanisms by which `f` was **not** pure — three
invisible inputs that made the same prefix compute different bits at
different times. Each violation is a separate defect; the first two produce
wrong-parity checkpoints, and wrong-parity checkpoints are what re-mount a
degenerate basin after every restart of the process that cached them —
until the chain is dropped.

### Defect A — non-finite q8_1 activation scales (fixed by this PR)

Three kernels emitted the q8_1 block pair `(d, sum)` unclamped
(`fused_gr.cu gr_q8_tail`, `verify_kernels.cu gdn_q8_1_store`,
`iq_kernels.cu s26_swiglu_q8_1_kernel`), bypassing the finite helper the
rest of the tree uses. A block with an overflowing `amax` emits a NaN scale;
NaN is not contained by the dot path; a fully non-finite logit row ends at
the sampler's all-NaN fallback — vocabulary id 0, the `!` face. Details and
diagnostics: `Q8_1_FINITE_DS.md`. This defect was environment-gated (fused /
S26 expert paths); deployments not enabling those paths were immune to
*face A* but fully exposed to defects B and C.

### Defect B — expert output depends on residency tier (the cross-session latch)

The sparse-expert tier migrates experts between GPU residency and the CPU
pool while serving. The two paths are not bit-identical: the GPU path packs
the activation `(d, sum)` pair as fp16 (rounding `d`), the CPU pool keeps an
fp32 scale. Same expert, same input, different bits depending on **where the
expert happened to live this second**. With the adaptive migrator running,
`f(prefix)` silently acquires wall-clock and access-history as inputs:
a checkpoint captured under one residency table restores to a different
number stream under another.

Confirmation (falsification): a state-parity probe hashing every tensor the
checkpoint restore path carries. Pre-fix, identical prefixes produced three
distinct state-hash sets depending on capture/restore history; with the
migrator disabled (`--adapt-swaps 0`, freezing the residency table for the
process lifetime), the same probe is byte-identical across fresh prefill,
checkpoint resume, and park/restore — and the cross-session latch stopped
reproducing. The freeze is a mitigation, not the remedy: it restores the
contract by giving up expert promotion. The engine-side remedy is to make
the **checkpoint identity tier-aware** — capture a fingerprint of the
residency table alongside the state tensors, and treat a mismatch at
restore as a cache miss (re-prefill on the existing miss path), so forks
are trusted exactly when they were computed under the table they will
restore into. A stronger alternative is tier-invariant numerics (one
quantization of the activation scale across both paths), but the two
formats compute on different hardware with different dot-product shapes,
so the fingerprint route is the one that preserves the migrator. Either
way, plus verifying all state tensors on restore, not only the token
prefix.

### Defect C — the draft policy learns from wall time (mid-run kicks)

The speculative window shape is chosen by a process-scoped policy whose cost
estimates are learned from **measured round times**. Wall time is not a
function of the prefix, so two identical requests arriving at different
moments can choose different window shapes; and the batched verify kernels
are not bit-invariant across window shapes, so window shape rides the output
bits. Greedy decoding amplifies a last-bit difference whenever the top-2
logits are near-tied, which is exactly what long generations hit.

Confirmation (falsification): identical greedy requests (temperature 0,
sampling ruled out with seed-pinned arms) of ≥100 generated tokens diverge
run-to-run with the suffix-draft source enabled, with per-round draft
acceptance counts varying; at depth ≤5 tokens they never diverge (no shape
decision has had a chance to differ). Disabling the suffix-draft source —
removing the policy's decision surface — makes five identical long requests
**byte-identical**, with byte-stable acceptance counts. The short-run/long-run
bisection and the acceptance-count scatter are the cheap field diagnostics.
This defect does not by itself explain the cross-session latch (the policy
dies with the process); it supplies the kicks that walk a trajectory into a
basin that defect B then makes sticky.

## 3. Mechanism, in one paragraph

The sampler turns non-finite input into vocabulary id 0 and turns ulps into
coin flips at near-ties. Defect A manufactures non-finite activations from
ordinary overflow; defects B and C make the state cache's `f(prefix)` impure,
so cached checkpoints fork the wrong continuations and inherit basins across
sessions, while the drafting policy kicks trajectories toward them. Restarts
cure because they drop the carried chain; new sessions do not escape because
they re-fork the same poisoned root. The unifying invariant for engines that
cache state: **any component that chooses *how* to compute — expert
residency, batch/window shape, drain order, quantization tier — is part of
the output function unless its numerics are provably shape-invariant. If the
invariant cannot be proven, the choice must be frozen for the process
lifetime, keyed into the cache identity, or shared as one numeric image.**

## 4. Diagnostic protocol (field checklist)

1. On onset, **preserve the episode log before restarting anything.** Record
   onset turn and the first degenerate turn.
2. Restart the engine and re-read the same prompt. Clean ⇒ carried state
   (this family). Still dirty ⇒ prompt/model problem (different tree).
3. Identify the face: id-0 (`!`) runs ⇒ check for unclamped q8_1 emit sites
   and any NaN sentinel in the sampler (defect A). Small literal-token
   collapse with structure ⇒ suspect state parity (B) and kicks (C).
4. Cross-session: open a brand-new session mid-episode. Affected ⇒ shared
   root checkpoint is the carrier (B). Cure by dropping the conversation
   cache/checkpoint chain, then freeze the migrator for verification.
5. Parity probe: hash the restored state tensors for one fixed prefix via
   (a) fresh prefill, (b) checkpoint resume, (c) park/restore. Divergent
   hashes ⇒ defect B class. Byte-identical but long greedy requests still
   fork ⇒ defect C class: bisect by generation depth and read per-round
   draft acceptance counts; rule out sampling with explicit `temperature=0`
   and a seed-pinned arm first.
6. Guard placement: text-level repetition detectors are a safety net, not a
   fix. They catch faces, not causes, and a cause fixed upstream makes them
   quietly idle — which is the correct end state.

## 5. What this PR changes, and what it deliberately does not

Changes: the three emit sites clamp through `q8_1_ds()`, matching the rest
of the tree (defect A). No behavior, tuning, or policy changes are included.

Deliberately excluded (deployment-side, not engine fixes, listed so
maintainers can route them properly): freezing the expert migrator via an
existing flag (defect B mitigation within a process lifetime); disabling the
suffix-draft source (defect C mitigation at a decode-speed cost); serve-layer
repetition detectors and a state-drop verb for in-place unlatching. The
engine-side remedies these stand in for — a shared numeric image across the
residency split, checkpoint identity keyed to the residency table (a
fingerprint at capture, a cache miss on mismatch), shape-invariant verify numerics
or a reproducible policy input, and full-tensor restore verification — are the
recommendations this RCA exists to support.
