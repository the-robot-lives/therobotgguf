# 004 — Settle-to-Answer: Parallel Iterative-Refinement Generation

**Status:** proposal
**Primary targets:** faster output, gestalt cognition, compute proportional to difficulty
**One-line thesis:** Abolish token-by-token emission. Draft the entire answer in parallel, then *settle* it to a fixed point through top-down feedback iterations — §C's FeedbackController becomes the decoder. K settling steps over N tokens replaces N sequential steps, and K ≪ N.

---

## 1. The bottleneck we're attacking

Autoregressive decoding is a hard sequential chain: N output tokens = N full forward passes that cannot be parallelized across the answer, each one memory-bandwidth-bound on weight and cache reads. Every other part of the stack parallelizes; generation does not.

It is also, per the notes, the *cognitively wrong shape*. The young/old-woman illusion doesn't resolve left-to-right — perception is a parallel relaxation into an attractor, with one interpretation locking in globally. The notes' settling story (§2 of notes, §C of the plan) describes a decoder, and this proposal takes it up on that.

## 2. The overhaul

- **Canvas, not stream.** Start from a fully-masked (or noised) answer canvas of estimated length. Each refinement step processes **all positions in parallel**, committing high-confidence positions and leaving low-confidence ones molten.
- **§C is the sampler.** Each denoising step = one top-down settling iteration: high-level conclusions re-prime low-level token choices. Stop when change < ε ("settled = done") under a step cap — the exact guardrails planning §C already specifies.
- **Adaptive compute.** Easy prompts settle in 2–4 steps; hard ones take 12–16. Compute is proportional to difficulty — the adaptive-compute property AR decoding structurally can't have. The salience gate (§F) can demand extra settling rounds on high surprise ("that surprised me — re-check"), and **m** schedules the paranoia: an anxious modulator state runs more re-check iterations. The notes' anxiety loop, implemented as a sampler schedule.
- **Bistability becomes observable.** Two valid completions = two attractors; you can watch the canvas commit to one. Hypothesis 2 turns from an architectural claim into a decoding-time measurement.
- **Revision is generation.** The canvas is non-causal, so infilling and editing are the same operation as writing — thought as redrafting, free of charge.

## 3. Training

- **Masked-diffusion LM objective:** all positions supervised per example (BERT-density, no causal-mask waste), no exposure bias, and the training-time task matches the inference-time process — unlike AR, where teacher forcing and free-running diverge.
- **Consistency distillation** compresses a 64-step settling schedule into 4–8 steps after the fact. Note the hook: distilling slow deliberate settling into a fast reflex is *habit formation*, and Proposal 005's consolidation daemon is the natural home for running it continuously.

## 4. What it buys (honest arithmetic)

For a 500-token answer: AR = 500 sequential forwards; settling = 8–16 parallel passes → 30–60× fewer *sequential* steps. Each settling pass costs more than a one-token AR step (it touches the whole canvas), so the honest net at batch 1 is a **3–10× latency cut**, more after distillation. Secondary win: a single request saturates the GPU (parallel across positions) where AR decoding starves it — better hardware economics at low batch sizes, which is exactly the self-hosted single-user regime.

## 5. Mapping to planning.md components

| Component | Role in this proposal |
|---|---|
| §A modulator **m** | Schedules settling depth (anxious → more re-checks) |
| §B leaky state | Persists across settling steps; provides the *un-commit* escape hatch (the hose can stop being a snake mid-settle) |
| §C feedback | *Is* the decoder — FeedbackController graduates from component to product |
| §D bottlenecks | Readable mid-settle: watch interpretation form before tokens freeze |
| §E shims | Can edit the molten canvas before lock-in — steering during settling is cheaper and more surgical than steering a committed stream |
| §F memory | Surprise triggers extra settling rounds; settled summaries are what gets written |

## 6. Falsifiable tests

1. **Parity band:** matched-parameter quality vs the AR baseline; accept a stated initial gap (e.g., ≤5% on the eval suite) and track it per milestone.
2. **Difficulty proportionality:** steps-to-settle distribution correlates with independent prompt-difficulty measures. This is the headline claim — if compute doesn't scale with difficulty, the proposal loses its distinctive value.
3. **Bistability:** ambiguous prompts produce bimodal commitment (one attractor per sample), not blended text. Measures Hypothesis 2 at the output layer.
4. **Batch-1 latency** vs AR on 200–1000-token answers, pre- and post-distillation.

## 7. Risks & mitigations

- **Quality gap vs AR** — the field's gap is closing fast (LLaDA, Dream reached AR-7B-class parity on many tasks in 2025) but it is real; keep an AR head as fallback and for speculative verification (settle-draft → AR-verify composes with speculative decoding).
- **Length prediction** (how big a canvas?) → train a cheap length head; over-allocate and let the canvas emit end-padding.
- **Oscillation between attractors** → step cap + EMA damping, per §C's existing guardrails; measure non-convergence rate explicitly.
- **GGUF/llama.cpp has no settling loop** → same custom-runtime bucket as Proposal 002; the two share an executor (iterate-until-quiet is one control structure implemented once).

## 8. Prior art to build on

Masked diffusion LMs (LLaDA, Dream 7B, MDLM line); deep equilibrium models; Jacobi/lookahead parallel decoding; consistency models & distillation; modern Hopfield networks (attention as attractor retrieval); speculative decoding (composable, not competing).

## 9. Fit with the milestone plan

Lands at Milestone 4 (feedback) but changes its emphasis: build the FeedbackController as a *decoder loop* from day one, with the bistable-percept test moved from a probe to an acceptance criterion. Milestone 1's GGUF spike should include a stub settling loop in the runtime so the export question (planning §8, last bullet) is answered before this proposal is load-bearing.
