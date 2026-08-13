# 005 — Frozen-Core Accretion: Training as Growth, Never Retraining

**Status:** proposal
**Primary targets:** collapse the *marginal* cost of training, continual learning without forgetting, offline consolidation
**One-line thesis:** Train a modest core once, cleave it (§D), freeze it permanently. Every subsequent capability arrives as an accreted module — shim (§E), memory (§F), modulator policy (§A) — trained against *recorded* interfaces, in parallel, at adapter cost. This scales the §7 "cleave and reinforce" recipe from a technique into the entire training economy.

---

## 1. The bottleneck we're attacking

The industry's answer to every capability gap is to retrain the monolith end-to-end: O(model × data) each time, catastrophic forgetting managed by re-mixing all previous data, zero reuse across runs. Biological systems never do this: a brain does not re-derive vision to learn a fact; learning is local, incremental growth plus offline consolidation, on top of circuitry that stays put for decades.

## 2. The overhaul

1. **Build the core once, cleave densely.** Bottlenecks (§D) every few blocks, generous attribute coverage. Core quality gates everything downstream — this is where the one-time spend goes.
2. **Record.** Run the corpus through the frozen core; store bottleneck activations. This *dataset of interfaces* becomes the training substrate for all future modules.
3. **Accrete.** New capabilities are modules trained **against the recordings — the core isn't even loaded during training.** Consequences, in order of importance:
   - **Adapter-scale cost:** millions of trainable parameters, not billions.
   - **Embarrassingly parallel:** N candidate modules train simultaneously against the same cached activations with zero interference. Population-based search over module architectures and hyperparameters becomes routine; admit the winner.
   - **Training hardware ≠ inference hardware:** recordings train modules on modest GPUs or the CPU fleet — the self-hosted cluster becomes a capability factory without ever holding the full model in memory.
4. **Admission control.** Verify against the live core before registry entry: selectivity score (change the target, hold everything else), regression suite vs the oracle, composition check against already-admitted shims. The registry (name, target bottleneck, verified effect, selectivity) is the notes' "$5.99 module" marketplace with QA attached.
5. **Consolidation daemon ("sleep").** Salient episodic memories (§F) replay offline to distill recurring in-session adaptations into permanent shims. Slow deliberate behaviors — e.g., Proposal 004's multi-step settling on a recurring task class — distill into fast reflex shims. Deliberate → automatic is habit formation; episodic → parametric is complementary-learning-systems theory, implemented literally as a cron job.
6. **Growth, not just adaptation.** When shims against existing bottlenecks hit a residual-error floor, graft *new blocks* at cleave points (progressive-network style) — capacity expansion with old weights untouched.

## 3. What it buys (honest arithmetic)

- Marginal capability cost drops from full-run scale (GPU-months) to adapter-run scale (GPU-hours): the ratio is roughly trainable-params × data-passes, i.e., **10³–10⁴× cheaper per capability** once the core exists.
- **Forgetting is structurally impossible** — frozen weights cannot drift. (Verify anyway; interference can still arrive through composition.)
- Parallel capability development across teams or agents with no merge conflicts, because the interfaces are versioned contracts.
- **Inference gets faster as capability grows,** not slower: route and load only task-relevant shims (module-granular mixture-of-experts). The core stays resident and hot; shims stream in and out. Per-request compute tracks *needed* capability, not *accumulated* capability — the exact opposite of monolith scaling.

## 4. Mapping to planning.md components

| Component | Role in this proposal |
|---|---|
| §7 recipe | *Is* this proposal's kernel, promoted to the whole methodology |
| §D bottlenecks | The recorded, versioned contracts everything trains against |
| §E shims | The unit of capability and of commerce |
| §F memory | The fast store; consolidation's raw material |
| §A modulator | Modulator *policies* are themselves accreted modules |
| Proposal 001 | Same machinery — module-local training against fixed interfaces |
| Proposal 004 | Settling behaviors are what consolidation distills into reflexes |

## 5. Falsifiable tests

1. **Capability-add cost:** a new verified capability lands at ≤1% of core-training FLOPs with selectivity above the admission bar.
2. **Zero forgetting:** core-task metrics before/after admitting 50 modules — must be bit-identical on the frozen paths, and within noise end-to-end with routing active.
3. **Consolidation:** an in-session adaptation (Hypothesis 4's one-shot behavior change) that recurs across sessions gets distilled by the daemon and survives episodic decay — the behavior persists after the memory that taught it has faded. That is the strongest "faintly mind-like" demo in the entire proposal set.
4. **Parallelism:** N modules trained concurrently show linear throughput scaling against the shared recording store.

## 6. Risks & mitigations

- **Core ceiling.** A frozen core caps reachable representations. Mitigations: cleave generously now (cheap insurance), keep the block-growth path warm, and accept that a *generational* re-core every year or two is an event, not a failure — with the registry re-verified against the new core. (This is the Accord's shadow-deployment consensus-upgrade protocol, applied to the substrate.)
- **Recording storage.** Bottleneck activations across a corpus are large → store only §D slices (small by design), subsample, quantize; recordings are derived data and can be regenerated.
- **Interface drift.** Never patch the frozen core in place; version it. A patched core silently invalidates every recording and every admitted module — the one operational sin this architecture cannot forgive.
- **Shim conflict.** Composition order matters once shims stack → registry declares dependencies/conflicts; admission includes compose-verification against the active set, and the verified pairs are recorded.

## 7. Prior art to build on

PEFT/LoRA ecosystems and adapter hubs; adapter composition and model merging; progressive networks (Rusu et al.); Net2Net function-preserving growth; complementary learning systems (McClelland, O'Reilly); wake-sleep and experience replay; knowledge distillation; mixture-of-experts routing; activation caching from the probing/interpretability literature.

## 8. Fit with the milestone plan

This formalizes Milestones 6–8 and extends them past the finish line: Milestone 6's shim registry becomes admission control; Milestone 7's memory gets the consolidation daemon; Milestone 8's composition demo becomes the steady-state development loop. The prerequisite investment is all in Milestone 5 — cleave coverage and recording infrastructure — which this proposal argues should be over-built relative to the plan's current sizing, because every later economy compounds on it.
