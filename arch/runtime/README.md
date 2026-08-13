# therobotgguf — Runtime & Conversion Plan

**Status:** plan / pre-implementation
**Scope:** (1) converting *existing* LLM checkpoints into models that run on the therobot runtime; (2) building that runtime as a deep fork of llama.cpp.
**Companions:** [`conversion-pipeline.md`](conversion-pipeline.md) · [`llamacpp-extensions.md`](llamacpp-extensions.md) · [`gguf-extension-spec.md`](gguf-extension-spec.md)
**Grounding:** [`../../planning.md`](../../planning.md) components §A–§F and [`../proposals/`](../proposals/) 001–006.

---

## 1. Decisions taken (settled, not open)

| Question | Decision | Consequence |
|---|---|---|
| Conversion/training stack | **Python-first** — HF Transformers + PEFT-style training, `gguf-py` for emission | Elixir Nx/Axon remains the ground-up research stack per planning.md §6; it is *out of scope* for the retrofit path |
| First donor family | **Small Llama/Qwen-class transformer (1–4B)** | Exercises the full retrofit path (state must be grafted on); Mamba/RWKV-class is the second target and gets §B "for free" |
| llama.cpp integration | **Deep fork, new architecture family** | Single-file extended GGUFs, first-class ops and state. We accept the rebase burden and mitigate it structurally (see llamacpp-extensions §2). Extended files do **not** load in stock llama.cpp; a `strip` tool downgrades them for interop |
| v1 scope | **All capability tiers, planned as stages** | No single-tier v1; the roadmap below sequences L0→L5 with a hard gate at each stage |

## 2. Capability levels

A converted model and the runtime negotiate features via the GGUF spec (`therobot.features`). Levels are cumulative summaries, not a strict lattice — L3/L4/L5 depend on L1–L2 but not on each other.

| Level | Name | planning.md / proposal | What it adds |
|---|---|---|---|
| **L0** | Passthrough | — | Donor converts and runs **identically** to stock. The permanent parity gate |
| **L1** | Taps + shims | §D, §E | Typed bottleneck read-out during decode; probe heads; shim modules (slice-scoped adapters) hot-loaded at runtime |
| **L2** | State + modulator | §B, §A | Grafted leaky temporal state banks; modulator bus **m** with FiLM gating; session state = a checkpointable "mind" (003 §4) |
| **L2.5** | Episodic memory | §F | Salience-gated content-addressed store; recall feeds **m** and shims. Runtime-side; needs L1 taps + L2 modulator |
| **L3** | Delta inference | 002 | Change-triggered execution: thresholds, held outputs, block-granular skip, heartbeat sweeps |
| **L4** | Settling decoder | 004, §C | Canvas-based parallel iterative refinement; **m**-scheduled settling depth |
| **L5** | Accretion serving | 005, 006 | Shim registry + admission metadata, per-request routing, hot-swap, consolidation-daemon interfaces. 006 stays training-side — exports are always recompacted static graphs (006 §8) |

## 3. The two workstreams, and how they interlock

**Workstream C — conversion** (`conversion-pipeline.md`): a Python pipeline `robotgguf` that takes a HF checkpoint through: ingest → record activations → cleave (train probes, pick bottlenecks) → graft (state + modulator, function-preserving init) → calibrate (delta thresholds) → shims/salience → export → verify. The core principle is proposal 005's: **the donor core is frozen from the moment of recording**; every new parameter trains against recorded interfaces or with the core frozen, and every graft is zero-effect at insertion (006's nascent-unit trick).

**Workstream R — runtime** (`llamacpp-extensions.md`): a deep fork of llama.cpp registering a `therobot` architecture family that wraps the donor's base graph with extension insertion points, plus a shared "iterate-until-quiet" executor serving both 002 and 004 (the proposals README synergy: one custom-runtime component, not two).

They meet at the **GGUF extension spec** (`gguf-extension-spec.md`) — the single contract both sides implement. Spec version bumps are the only permitted form of interface change (005 §6: never patch contracts in place).

## 4. Staged roadmap

Each stage has a Definition of Done that is a runnable demo + a regression gate. A stage does not start until the previous stage's gate is green. Effort tags are relative (S=small, M=medium, L=large, XL=research-grade).

| Stage | Delivers | Conversion work | Runtime work | Definition of Done |
|---|---|---|---|---|
| **S0** Bootstrap (M) | L0 | R0 ingest, R7 export (passthrough), R8 harness skeleton | E0 fork, E1 spec loader | Qwen/Llama 1–4B donor → therobot GGUF; fork logits match stock llama.cpp on the same weights ≤1e-4; upstream test suite green in fork |
| **S1** Taps + shims (M) | L1 | R1 record, R2 cleave, R5 first shims | E2 taps, E3 shim engine | Live attribute read-out during decode; one shim shifts its target attribute past the selectivity bar while parity holds with shims off |
| **S2** State + modulator (L) | L2 | R3 graft + graft training | E4 state banks, FiLM, session serialization | Priming demo: induce then relax a false-positive bias via **m** (planning M3); checkpoint/restore resumes mid-session state; zeroed grafts = L0 parity |
| **S3** Episodic memory (M) | L2.5 | R5 salience calibration, summary heads | E5 memory store | One-shot in-session behavior change that decays as designed (Hypothesis 4) |
| **S4** Delta inference (L) | L3 | R4 threshold calibration | E6 delta executor | ≥2× FLOP cut at <1% quality delta, batch-1 CPU streaming; p50/p95/p99 vs dense reported; bounded divergence with heartbeat |
| **S5** Settling decoder (XL) | L4 | R6 settle-donor track | E7 settle decoder | Settle vs AR batch-1 latency on 200–1000-token answers; steps-to-settle correlates with difficulty (004 test 2) |
| **S6** Accretion serving (M) | L5 | registry tooling, admission suite | E8 routing/hot-swap | 10 shims hot-loaded and routed per-request; core-path outputs bit-identical before/after (005 test 2) |

Notes on ordering: S3 is deliberately early (cheap, pure-runtime, high demo value). S4 and S5 are independent after S2 and can swap or parallelize. S5's donor question is resolved by *starting from an existing diffusion checkpoint* (Dream/LLaDA-class) run through the same pipeline, rather than converting the AR donor — converting the AR donor to a masked-diffusion objective is a later, compute-heavy experiment (see conversion-pipeline §R6).

## 5. Verification philosophy (applies to every stage)

1. **Parity is a permanent invariant, not a milestone.** With every extension disabled — or its parameters at function-preserving init — the fork must reproduce stock logits on the donor. Every stage re-runs the parity gate.
2. **Toggle matrix.** Each feature independently on/off; quality (perplexity + small eval suite) and latency recorded per cell.
3. **Behavioral probes over benchmarks** for the novel mechanisms: the priming/false-positive induction (002 test 4), memory decay curve (Hypothesis 4), settle-step difficulty correlation (004 test 2). These are the tests that justify the project; a failed one is a finding that branches the plan (planning.md's rule).
4. **Honest latency**: percentiles, never means (002 §4).

## 6. Top risks

| Risk | Mitigation |
|---|---|
| Deep-fork rebase burden | All extension code in new `llama-robot-*` files; insertions fenced with `ROBOT-EXT` markers; stock arches untouched (fork is a strict superset — upstream tests must stay green); scheduled upstream syncs |
| Extended GGUFs unreadable by stock tooling | `robotgguf strip` downgrades to a stock base-arch file; document loudly |
| Text-domain attribute choice for §D bottlenecks (disentanglement risk, planning §8) | Start with coarse, highly decodable attributes; report selectivity honestly; treat cleave-site selection as measured, not declared |
| Settling adaptation compute (004) | Ship settle support against existing diffusion checkpoints first; AR-donor conversion deferred |
| Delta mode vs batched serving | v0 delta executor is batch-1 streaming only (002's stated CPU/self-host target); batching is future work |
