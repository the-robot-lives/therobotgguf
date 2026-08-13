# therobotgguf — Architecture Proposals

Six candidate overhauls to how models are **trained** and **run**, each attacking a different serialization or inefficiency in the standard transformer pipeline, and each grounded in the components already specified in [`../../planning.md`](../../planning.md) (§A modulator, §B leaky state, §C feedback/settling, §D typed bottlenecks, §E shims, §F episodic memory). None requires a scientific miracle; every mechanism has working prior art. As with planning.md, the novelty is the composition and the recipe.

## The six

| # | Proposal | What it kills | Primary win | Biological motivation | planning.md anchor |
|---|---|---|---|---|---|
| [001](001-sundered-backprop-local-learning.md) | **Sundered Backprop** | The global backward pass | Training wall-clock via module-parallel local learning | Local, region-wise learning; no global error broadcast | §D cleave points as training contracts; §C feedback as teacher |
| [002](002-event-driven-delta-inference.md) | **Event-Driven Delta Inference** | The dense per-token forward pass | 3–10× inference FLOPs/latency via change-triggered compute | Sparse asynchronous firing; adaptive thresholds | §A/§B thresholds + leaky state, made the execution model |
| [003](003-multi-timescale-state-backbone.md) | **Multi-Timescale State Backbone** | Attention-over-history & the KV cache | O(L) training, O(1)-per-token decoding, flat latency at any context | Nested timescales; executive vs background | §B promoted to backbone; §A absorbed as the glacial bank |
| [004](004-settle-to-answer-parallel-generation.md) | **Settle-to-Answer Generation** | Token-by-token autoregression | 3–10× output latency via parallel iterative refinement; compute ∝ difficulty | Gestalt settling; attractor lock-in | §C FeedbackController becomes the decoder |
| [005](005-frozen-core-accretion.md) | **Frozen-Core Accretion** | Retraining the monolith | 10³–10⁴× cheaper marginal capability; zero forgetting | Growth, habit formation, offline consolidation | §7 recipe promoted to the whole training economy |
| [006](006-unit-lifecycle-dormancy-and-spawning.md) | **The Unit Lifecycle** | Fixed topology (static width) | Capacity that tracks the task; continual learning without forgetting | Growth, pruning, dormancy, offline consolidation | 005 at unit granularity + continuous; §A gates spawn rate, §B carries age/vitality |

## Coverage of the brief

- **Faster training:** 001 (wall-clock parallelism), 003 (O(L) sequence scaling), 005 (marginal-cost collapse), 006 (adapter-cost spawns, right-sized nets)
- **Parallelization like real brains:** 001 (learning), 002 (execution), 005 (capability development), 006 (structural turnover)
- **Arrow of time:** 003 (constitutive state), 002 (history-dependent firing), 005 (consolidation over sessions), 006 (topology encodes history)
- **Faster output:** 002 (sparse compute), 003 (cache-free decoding), 004 (parallel generation), 005 (route-only-what's-needed), 006 (right-sizing reclaims idle capacity)
- **Growth & turnover (node dormancy/creation):** 006 (unit lifecycle), with 005 as its coarse module-level counterpart

## They compose: the full stack

The proposals are designed to be mutually compatible; the maximal composite is one coherent machine:

```
003 multi-timescale SSM core          ← the substrate (time arrow built in)
  trained via 001 local-learning mesh ← modules learn in parallel at cleave points
  executed via 002 event-driven deltas← only changed paths compute
  decoding via 004 settle-to-answer   ← whole-thought refinement, §C as sampler
  lifecycle via 005 frozen-core accretion ← capabilities accrete; offline consolidation distills
    at unit grain, 006 lifecycle       ← units are spawned, go dormant, and retire within modules
```

Notable synergies:
- **002 + 003:** slow state banks change rarely → almost never fire → near-free background processing.
- **004 + 005:** consolidation distills slow settling into fast reflex shims (deliberate → automatic).
- **001 + 005:** the same module-local training machinery serves both; 001's contract-stability test is 005's admission gate.
- **002 + 004:** one shared "iterate until quiet" runtime executor covers both custom-loop needs (and concentrates the GGUF custom-runtime risk from planning §8 into a single component).
- **006 + 002:** event-driven execution is what makes over-capacity affordable — idle/dormant masked slots don't fire, so they cost ~nothing until reclaimed. Load-bearing for each other.
- **006 + 005:** same offline consolidation cycle runs both — memory consolidation, shim distillation, *and* the unit census (spawn/retire/recompact) in one pass. 006 is 005's binary freeze-and-graft made continuous and fine-grained.

Known tension to watch: 001's local losses vs 004's global denoising objective — a module optimizing a local target mid-settle may fight the canvas-level fixed point. Resolution order: get 004 working with standard training first; introduce 001 for shim/module training (its natural home) before attempting it on the settling core.

## Suggested sequencing vs the milestone plan

| Milestone (planning §6) | Proposal that lands there |
|---|---|
| M1 baseline + GGUF spike | Stub the custom runtime loop (002/004's executor) into the export test |
| M2 temporal state | Run **003** as a bake-off: transformer+LeakyState vs SSM backbone, same params |
| M3 modulator | **002** pilot (feed-forward blocks only) once **m** exists |
| M4 feedback | Build FeedbackController as **004**'s decoder loop from day one |
| M5 bottlenecks | Over-build cleave coverage + recording infra per **005** |
| M6 shims | **001** pilots as the shim trainer; registry becomes **005** admission control |
| M7 memory | Add **005**'s consolidation daemon behind the salience gate; fold **006**'s census into the same offline pass (offline-only turnover first) |
| M8 integration | The composite stack above *is* the integration demo; **006** masked fast path + continual-learning eval once **002**'s executor exists |

Each proposal carries its own falsifiable tests; per planning.md's rule, a failed test is a finding that branches the plan, not a reason to paper over it.
