# 003 — Multi-Timescale State Backbone (The Clockwork Core)

**Status:** proposal
**Primary targets:** the arrow of time as the core primitive, linear-time training, O(1)-per-token cache-free decoding
**One-line thesis:** Promote §B from add-on to backbone. Replace attention-over-history with nested banks of leaky state at token / phrase / scene / session timescales; keep attention only as a narrow local window, and let episodic memory (§F) handle verbatim long-range recall.

---

## 1. The bottleneck we're attacking

Attention gives the transformer a perfect verbatim buffer and charges for it twice: O(L²) compute during training, and a KV cache that grows O(L) during decoding — making generation memory-bandwidth-bound and context length a cache-size problem.

Biological memory holds no verbatim buffer; it runs cascades of decaying traces at multiple time constants (milliseconds → seconds → minutes) plus content-addressed episodic recall when verbatim detail is actually needed. That is *exactly* §B + §F — the plan already contains the replacement; this proposal commits to it.

## 2. The overhaul

- **Backbone block = selective/diagonal SSM** (Mamba-class) with learned per-channel decay α, grouped into explicit **banks**:
  - **fast** (α≈0): token-rate, syntax and surface
  - **mid**: phrase/sentence-rate, local semantics
  - **slow**: paragraph/scene-rate, discourse and situation
  - **glacial**: session-rate — and here is the unification: the glacial bank *is* the modulator **m** (§A) and the write-source for episodic memory (§F). §A stops being a separate module; it's the slowest, lowest-dimensional stripe of the same state hierarchy.
- **Local attention window** (e.g., 128 tokens) for the things SSMs are genuinely worse at: verbatim copying, local induction.
- **Long-range verbatim recall via §F**, content-addressed — the episodic store replaces the KV cache's job at distance. Recall injects into the state banks rather than into an attention matrix.
- **Cross-timescale routing:** slow banks modulate fast processing (FiLM-style, reusing §A machinery). The Apr-9 "polite stranger" slow-burn is the design signature: the slow bank accumulates evidence across many tokens until it flips the fast bank's interpretation.

## 3. Training: linear and chunkable

- The recurrence trains as an **associative scan** — parallel across the sequence, O(L) not O(L²).
- **Chunked training with carried state:** train on segments while carrying state between them → effectively unbounded context at fixed memory. Long-horizon behavior is trained through *state*, not through ever-longer attention windows. This changes the economics of context: cost per token of context goes to a constant.

## 4. Inference: flat forever

- State is fixed-size → per-token compute and memory are **constant in context length**. No cache growth, no bandwidth cliff; latency at token 1,000,000 equals latency at token 100.
- Small state means **cheap mind-checkpointing**: snapshot the state banks = snapshot working memory. Forking, rollback (the Accord's Article IV), and session migration become tensor copies, not context-window replays.

## 5. What it buys (honest arithmetic)

Versus a transformer at context L: training O(L²) → O(L); decoding per token O(L) memory traffic → O(1). These are the published Mamba-family wins; we inherit them rather than claim them. The residual question is quality at fixed parameters, which is exactly Hypothesis 1's test.

## 6. Mapping to planning.md components

| Component | Role in this proposal |
|---|---|
| §A modulator **m** | The glacial bank — absorbed, not bolted on |
| §B leaky state | *Is* the backbone |
| §C feedback | Top-down enters as state-conditioning between settling passes |
| §D bottlenecks | Cleave *state banks*, not just activations — state slices are more temporally stable interfaces for shims |
| §E shims | Read/write state slices; a shim edit persists naturally via decay dynamics |
| §F memory | Replaces long-range attention; the slow→glacial pipeline feeds its writes |

## 7. GGUF / export

Mamba/RWKV-family architectures already have llama.cpp support. Of the five proposals this one has the shortest path to the project's namesake export format — a real strategic argument for making it the backbone and letting the other proposals ride on it.

## 8. Falsifiable tests

1. **Hypothesis 1, directly:** temporal-integration tasks at fixed parameter budget vs the Milestone-1 transformer baseline.
2. **Needle-in-haystack with memory assist:** state-only vs state+§F recall vs full attention — quantify what episodic recall recovers of what the cache gave up.
3. **Latency flatness:** per-token latency vs context length; the curve should be flat where the transformer's bends.
4. **Slow-burn task:** evidence distributed across a long sequence must flip the interpretation late — the multi-timescale variant should beat both a single-α SSM and the transformer at equal parameters.

## 9. Risks & mitigations

- **Verbatim recall weakness** of pure SSMs → the local window + §F hybrid is the mitigation; tune the ratio empirically (Jamba-style hybrids are the precedent).
- **EXLA scan kernels** — planning §8 already flags associative-scan custom-call work; Milestone 2 is the feasibility gate and this proposal raises its priority.
- **State interference over very long sessions** → glacial-bank consolidation into §F (offload old context to memory), which dovetails with Proposal 005's consolidation daemon.

## 10. Prior art to build on

S4/S5, Mamba 1/2 (selective state spaces); RWKV; Griffin/Hawk; Jamba-class hybrids; clockwork and multi-timescale RNNs; hierarchical SSM timescale analyses; memory-augmented transformers (RETRO) for the §F recall pattern.

## 11. Fit with the milestone plan

This *is* Milestone 2 taken seriously: instead of adding `LeakyState` to a transformer, make it the block and demote attention to a windowed assist. Recommend running Milestone 2 both ways (transformer+leak vs SSM-backbone) at the same parameter budget — the comparison is cheap at small scale and settles the backbone question with data before Milestones 3+ build on it.
