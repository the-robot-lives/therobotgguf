# llama.cpp Extensions — The therobot Runtime (Deep Fork)

**Status:** plan
**Decision:** deep fork with a new architecture family baked in (single-file extended GGUFs, first-class ops/state), *not* a sidecar/hook layer. We accept the rebase cost and pay it down structurally (§2).
**Location:** `projects/therobotgguf/runtime/llama.cpp` (git subtree, per repo convention).

---

## 1. Architecture family, not per-donor architectures

Register **one** new arch, `LLM_ARCH_THEROBOT` (`general.architecture = "therobot"`), with `therobot.base_architecture` naming the wrapped donor family (`llama`, `qwen2`, `mamba`, …). The therobot graph builder dispatches to the base family's builder through explicit **insertion points** and adds the extension subgraphs. This keeps the donor surface open-ended without an enum explosion, and keeps stock arches byte-identical to upstream (the superset invariant below).

## 2. Fork hygiene (the rebase-burden mitigation)

- **Superset invariant:** stock architectures and stock files behave identically to upstream; the upstream test suite must stay green in the fork at all times. CI runs both stock-model logit parity and upstream tests on every change.
- **New code in new files:** `src/llama-robot-*.{h,cpp}`, `ggml/src/ggml-robot*.{h,c}` (if any custom kernels), `examples/robot-*`, `tools/robot-*`.
- **Fenced insertions:** every touch inside an upstream file is wrapped in `// ROBOT-EXT-BEGIN(<id>)` / `// ROBOT-EXT-END` markers, and `docs/robot/patch-points.md` enumerates them — rebases become a mechanical checklist.
- **Scheduled upstream syncs** (roughly monthly, or on-demand for features we need); each sync is a PR whose only content is the rebase.

## 3. Work packages

### E0 — Fork bootstrap
Subtree add; monorepo build (CMake, Metal on the mac, CPU/CUDA for the cluster); CI job with the parity + upstream-test gates. Deliverable: fork builds, stock Qwen GGUF produces logits ≤1e-4 from upstream llama.cpp.

### E1 — Spec loader
Implement [`gguf-extension-spec.md`](gguf-extension-spec.md): `therobot.*` KV parsing into `llama_robot_hparams`, extension tensor mapping, feature negotiation (refuse unknown *required* features, ignore unknown optional ones), spec-version check. Deliverable: an L0 file loads, runs via the base builder untouched, and `tools/robot-inspect` dumps the extension manifest.

### E2 — Bottleneck taps (§D)
At each declared cleave point the graph builder marks the slice as a graph output (`ggml_view` → `ggml_cont` → named output tensor) — deterministic and cheap, no eval-callback fragility. Public API:

```c
int32_t llama_robot_tap_count(const llama_model *);
bool    llama_robot_tap_read(llama_context *, int32_t tap_id, float *dst); // last decode's slice
bool    llama_robot_probe_eval(llama_context *, int32_t tap_id, const char *attr, float *dst); // run probe head on demand
```

Probe heads execute only when asked. `llama-server` gets a `/robot/taps` endpoint mirroring this.

### E3 — Shim engine (§E)
Extends the existing adapter infrastructure (LoRA adapters + control vectors are already runtime-loadable — they are primitive shims). Additions: (a) **slice scoping** — a shim applies to a declared bottleneck slice, not a whole layer; (b) **mul+add** (FiLM-shaped) edits, not just additive steering; (c) **gating** — a shim can be conditioned on `m` or on a probe read; (d) **registry metadata** honored at load: `depends`/`conflicts` checked, admission scores surfaced. Shims are standalone GGUF module files (spec §4) and hot-load/unload per request.

### E4 — State banks + modulator (§B, §A)
- Reuse llama.cpp's **recurrent/hybrid memory** infrastructure (built for Mamba/RWKV/Jamba): a therobot-llama context owns the standard KV cache *plus* per-layer leaky-state slots and the per-sequence `m` vector. The EMA update and FiLM composition are plain ggml ops (`mul`/`add`/broadcast) inserted at the E1 insertion points — no custom kernels expected for v1.
- Modulator update runs as a tiny subgraph per decode step: pooled activations (+ E5 recall vector) → GRU/MLP → `m`, with per-channel decay toward baseline.
- **Session serialization:** extend `llama_state_get_data`/`set_data` to carry state banks, `m`, and the E5 memory store — this is 003 §4's mind-checkpoint (fork/rollback/migrate a session as a tensor copy).

### E5 — Episodic memory (§F)
Pure runtime component, CPU-side, no ggml changes: a per-context store of `{key, value, salience, timestamp}` where keys/values are memory-head projections of bottleneck summaries. Content-addressed read (cosine top-k) with recency weighting; salience-gated writes (surprise = logprob spike + `m` magnitude, quantile-normalized threshold from the converter). Recall injects into the modulator update and is exposed to gated shims. Persisted with session state; capacity-bounded with decay-based eviction.

### E6 — Delta executor (002)
v0 constraints: **batch-1, single-sequence streaming** (002's CPU/self-host target), **block granularity**. llama.cpp already builds the graph per ubatch, which is the opening: per token, compare each block's input against its held input (`|Δ| > θ_eff`, where `θ_eff = θ_base + fatigue − excitability(m)`); build the subgraph only for firing blocks; non-firing blocks contribute their **held output** from the per-block output cache. Dense **heartbeat sweep** every N tokens bounds drift. Activity statistics (which blocks fired, effective FLOPs) are recorded per token — the compute trace that 002's tests require. Channel-group granularity and batched delta are explicitly future work behind the same interface.

### E7 — Settling decoder (004)
Build on llama.cpp's existing diffusion-generation support (Dream/LLaDA-class): a `llama-robot-settle` tool + server mode implementing the canvas loop — draft all positions, iterate; per-position confidence commit; stop on `change < ε` under a step cap. therobot additions over stock diffusion decoding: **m**-scheduled settling depth (anxious → more re-check rounds), salience-triggered extra rounds, leaky state persisting across settling iterations (the un-commit escape hatch), taps readable mid-settle, shims editing the molten canvas. Optional AR-verify pass (speculative-style) using the same or a sibling model.

**Shared executor:** E6 and E7 are one control structure — *iterate until quiet* — implemented once (`llama-robot-executor`): a loop with a change metric, a threshold, a step cap, and per-iteration hooks. E6 instantiates it across tokens; E7 across settling steps. This concentrates the custom-runtime risk exactly as the proposals README prescribes.

### E8 — Accretion serving (005)
Registry file (index of admitted shim modules + admission scores + dependency graph); per-request routing (select/load shims by task tags before decode; core stays resident, shims stream); hot-swap without context teardown; export of session memory traces for the offline consolidation pipeline (consolidation *training* happens in the Python pipeline — the runtime only exports raw material and loads its distilled products). Zero-forgetting check baked into CI: core-path outputs bit-identical with routing active vs off (005 test 2).

## 4. Dependency graph

```
E0 ─▶ E1 ─▶ E2 ─▶ E3 ─▶ E8
             │
             └▶ E4 ─▶ E5
                 │
                 ├▶ E6 ─┐
                 └▶ E7 ─┴─ (shared llama-robot-executor)
```

Maps to roadmap stages: S0={E0,E1}, S1={E2,E3}, S2={E4}, S3={E5}, S4={E6}, S5={E7}, S6={E8}.

## 5. Testing

- **Parity gates** (every PR): stock GGUF in fork ≡ upstream; L0 therobot file ≡ its stripped stock twin; grafted-but-init file ≡ donor.
- **Toggle matrix** (nightly): each feature on/off × quality (perplexity + eval suite) × latency percentiles.
- **Behavioral probes** (per stage DoD): priming induce/relax with compute trace (002 test 4), memory one-shot + decay (Hyp. 4), delta bounded-divergence curve, settle difficulty-proportionality (004 test 2).
- **Upstream suite** green at all times (superset invariant).

## 6. Risks specific to the runtime

| Risk | Mitigation |
|---|---|
| Rebase conflicts in graph-builder internals (highest-churn upstream area) | Insertion points are few and fenced; if upstream refactors builders, the therobot builder wraps rather than patches (worst case: duplicate the base builder for the affected family) |
| Recurrent/hybrid memory API churn upstream | E4 isolates it behind `llama-robot-state.{h,cpp}`; only that file touches the memory API |
| Delta-mode correctness drift | Heartbeat sweeps + continuous bounded-divergence measurement (002 §10), delta off by default |
| GPU dislikes block-skip irregularity | v0 targets CPU streaming where 002 predicts the win; GPU delta is explicitly out of scope until measured |
| Settling quality gap vs AR | AR-verify fallback composes (004 §7); settle mode is opt-in per request |
