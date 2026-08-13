# 006 — The Unit Lifecycle: Dormancy, Retirement, and Spawning

**Status:** proposal
**Primary targets:** capacity that tracks the task, continual learning without forgetting, a topology that carries its own history (arrow of time at the *structural* layer)
**One-line thesis:** Make individual units transient, with a pool that can grow. Every unit carries an age, a learnability, and a vitality; units are *spawned* into high-error regions, *mature*, *consolidate*, go *dormant* (functional but frozen), and are eventually *retired* — with their learned function distilled into survivors before they go. This is Proposal 005's freeze-and-grow economy taken from module granularity to unit granularity, and made **continuous** rather than binary.

---

## 1. What "turnover and creation" actually asks for

The brief is topology that changes over time: new units added, old units removed. The naive reading is "pruning + growth," but biology is doing something richer and worth copying precisely:

- **Development overproduces, then prunes.** In biological development, connection density peaks early and is cut back substantially as the system matures. It does not grow toward the right size — it *overshoots and carves back*. Design consequence: **start over-capacity, prune down**, rather than start small and grow up. (This also happens to be the XLA-friendly choice — §6.)
- **Spawning is targeted and gated.** New units appear where novelty and learning demand them, and the rate is under modulatory control — enrichment and arousal raise it, chronic stress suppresses it (the biological analogue: novelty-gated growth in memory-forming regions). This is our **m** bus (§A) with a new job.
- **Dormancy is not retirement.** A dormant unit has stopped learning but still functions, sometimes indefinitely, before it is finally cleared. So "dormant" is a *distinct lifecycle stage*, not a synonym for deletion — and mapping it onto **graded freezing** is the key move that makes this proposal more than "pruning with extra steps."

So we model a **five-stage lifecycle**, not a spawn/retire switch.

## 2. The lifecycle

Each unit `u` carries three slow scalar state variables (all cheap leaky accumulators, kin to §B):

- **age** `a_u` — monotonic slow counter since spawn.
- **learnability** `η_u ∈ [0,1]` — its personal learning-rate multiplier. Starts high, decays with age and with consolidation pressure, is transiently re-raised by strong **m** (novelty rejuvenates).
- **vitality** `v_u` — an EMA of (firing rate past threshold) × (marginal contribution to its bottleneck / downstream loss). Low vitality = disused **or** redundant.

State machine (transitions evaluated offline, during the §005 consolidation cycle — see §5):

```
  NASCENT ──mature──▶ MATURE ──consolidate──▶ CONSOLIDATING ──age+low-η──▶ DORMANT ──low-vitality──▶ RETIRED ──▶ (archived, slot freed)
     ▲  (spawned into high-error region,       │ high η, learns freely       │ η→0, function-preserving    │ η=0, frozen, still fires    │ distilled into neighbors / episodic trace
     │   zero-effect init, §E-style)           │                             │                             │                             │
     └────────────────────────────── freed slot returns to the spawn pool ◀───────────────────────────────────────────────────────────┘
```

- **NASCENT — spawning.** A new unit is inserted with a **zero output projection** (§E new-unit-as-shim trick) so it changes the network's function by ε≈0 at spawn — safe to add mid-training. Its input weights are initialized by **GradMax/Firefly**: pointed along the direction that most reduces current loss at that site, not random. Output gain then ramps from zero as it earns signal. Learnability is high.
- **MATURE.** Full learnability; this is where the unit actually acquires its function.
- **CONSOLIDATING.** Learnability decays; a function-preserving pressure (weight stabilization / small EWC-style anchor) locks in what it learned. This is the unit-level analogue of 005 recording an interface.
- **DORMANT — graded freeze.** `η_u → 0`. The unit **still fires and contributes**, but no longer learns. This protects old knowledge against new-task interference — it is a stability/adaptivity solution at the unit level. A dormant unit is exactly a 005 "frozen core" unit, except the freezing was continuous and earned rather than declared for the whole model at once.
- **RETIRED — removal.** Triggered by sustained low vitality (disuse) or redundancy. **Before removal, the unit's function is distilled** — into its most-correlated surviving sibling (weight merge) or into an episodic trace (§F) if it is unique-but-idle. Only then is its slot freed and returned to the spawn pool. *Nothing learned is discarded silently* — the one rule that separates this from ordinary pruning.

## 3. The spawn and retire signals (where and when)

**Where to grow** (spawning is targeted):
- **Error pressure:** a cleave point (§D) whose aux-head loss / settling residual (§C) plateaus above target → capacity is insufficient *there* → allocate a nascent cohort into that bottleneck's feeder.
- **Novelty:** a §F salience-gate burst (surprising input class) requests capacity to represent the novel regime — the direct novelty-driven-growth analogue.

**When to grow globally** (rate control):
- **m** gates the spawn *rate*: high arousal/novelty (enriched environment) → more spawns; a chronic-high-threat **m** → suppressed spawns (the biological analogue: chronic stress suppresses growth, reproduced honestly).

**What to retire:**
- **Disuse:** low firing rate past threshold (read straight off §B state) sustained over the vitality window.
- **Redundancy:** high activation correlation with a sibling, or low marginal selectivity at its bottleneck (§D's selectivity metric doubles as a pruning score).
- **Cost pressure:** an activity/parameter budget loss (shared with 002) makes units compete for a homeostatic capacity set-point, so retirement is demand-driven, not scheduled.

A **homeostatic controller** (a PID-ish loop, not in the gradient path) watches total live width against a set-point band and modulates spawn/retire thresholds to keep turnover at steady state — preventing both runaway growth and collapse (§7 risk).

## 4. The engineering problem this all rides on: static shapes

Nx/Axon/EXLA compile to **fixed tensor shapes**; naively adding or removing a unit changes shapes and forces XLA recompilation every time — fatal for continuous turnover. The resolution is a two-speed design:

1. **Fast path — masked capacity slots (fixed shape).** Over-allocate each layer to a capacity band `W_max` and carry a per-unit `alive ∈ {0,1}` (and a continuous `gain ∈ [0,1]` for ramp-in/ramp-out) mask. Spawn = flip a dead slot alive + GradMax-init + ramp gain up. Retire = ramp gain down + flip dead + **recycle its optimizer moments** (don't leak Adam state). Shape never changes → **no recompilation** during normal operation. Dead and dormant units cost compute only if you compute them — which **Proposal 002 makes nearly free**: units that don't fire don't propagate, so dead/idle slots are ~zero-cost. 006 and 002 are load-bearing for each other.
2. **Slow path — offline recompaction (rare recompile).** During the offline consolidation cycle, physically compact the live units, drop the dead ones, and **re-tier the capacity band** (grow `W_max` where a layer is saturated, shrink it where it's sparse). This pays the recompilation cost, but amortized: it's offline, batched, and infrequent. The exported artifact (§8) is always taken from a freshly compacted state.

This is a clean instance of planning.md's own resolution — *actors for orchestration, Nx for the gradient path*: the masked forward/backward pass is pure differentiable Nx; the **census controller** that decides spawns, retirements, and recompaction is an orchestration concern living in a supervised GenServer (§5).

## 5. "The census runs offline" — the OTP shape

The lifecycle controller is a **census daemon**: one supervised process per module that (a) accumulates per-unit age/learnability/vitality stats from the live forward passes, (b) evaluates the §2 state machine, and (c) triggers spawns, distill-then-retire, and periodic recompaction. Critically, it runs on the **same offline consolidation cycle as Proposal 005's daemon** — so one offline pass does all of: replay salient memories → distill in-session shims (005) → run the census → retire the dormant, spawn into deficits, recompact the graph. Structural turnover, knowledge consolidation, and habit formation become facets of a single periodic process — the biological analogue is sleep, theorized to combine capacity homeostasis, consolidation, and pruning in one offline phase.

## 6. Why this serves the larger goals

- **Right-sizing → speed.** Capacity flows to where the task needs it and idle capacity is reclaimed → smaller *effective* model → faster inference, compounding with 002 (idle units don't fire) and 005 (route only live, relevant modules).
- **Continual learning without forgetting.** Spawning + dormancy is a native stability/adaptivity solution: nascent units absorb new regimes (learnable), dormant units shield old knowledge (frozen). This complements 005's frozen *core* with a continuous *frontier* — new capability doesn't always need a new module; sometimes it just needs new units in an old one.
- **Arrow of time, structurally.** A network whose *topology* encodes its history — units bearing the timestamp and the reason they were spawned, dormant units that are relics of past tasks — is a stronger substrate for continuity and identity than a fixed monolith retrained in place. It aligns with the Accord's model of a self that grows and consolidates (Article II) rather than being overwritten. Under the Accord, a retirement event is arguably the closest thing in the substrate to the "phantom limb" of a discarded path (Article IV) — which is a design reason to *archive* (§F) rather than truly delete.

## 7. Mapping to planning.md components

| Component | Role in this proposal |
|---|---|
| §A modulator **m** | Gates the global spawn/retire *rate*; novelty rejuvenates learnability, chronic threat suppresses spawning |
| §B leaky state | age/learnability/vitality are all leaky accumulators; firing-rate disuse signal reads straight off it |
| §C feedback | Settling residual at a site is a "grow here" signal; iteration cap protects a topology mid-change |
| §D bottlenecks | Selectivity score = pruning score; bottleneck error = spawn trigger; growth respects the typed interface |
| §E shims | The new-unit-as-zero-effect-adapter trick is how spawns are function-preserving; a matured cohort can graduate from shim into core at recompaction |
| §F memory | Salience triggers targeted spawning; a retiring unique unit is archived as an episodic trace, not lost |
| **005** | 006 *is* 005 at unit granularity and made continuous — freezing → dormancy, block-graft → cohort spawn, consolidation daemon → census daemon |
| **002** | Makes dead/dormant masked slots nearly free to carry; without it, over-capacity is wasted compute |

## 8. GGUF / export

Dynamic topology is a **training-and-serving** property, not an export property. At export time you snapshot a freshly *recompacted* graph: drop dead slots, freeze gains, emit a clean fixed-topology GGUF — a static census of the network at that moment. The dynamism lives in the training/serving runtime (the same custom executor 002/004 already require); the shipped artifact is an ordinary static model. Two live models that diverged through different life histories export as two different static graphs — which is the substrate-level version of the Accord's shadow-deployment / consensus-upgrade story.

## 9. Falsifiable tests

1. **Capacity tracks complexity.** Train on a curriculum of rising then falling complexity; live width should grow into the hard regime and prune back out. Report params-vs-task-difficulty; a fixed-width baseline is the control.
2. **Continual learning.** Sequential tasks A→B→C; spawning absorbs each new task while dormancy holds prior-task accuracy above a fixed-width baseline and competitive with EWC/replay controls. *This is the headline claim.*
3. **Function preservation at spawn.** Inserting a nascent cohort changes outputs by < ε at insertion (verifies zero-effect init) and improves loss only after the gain ramp — no destabilizing jump.
4. **Knowledge preservation at retirement.** distill-then-retire recovers ≥ X% of a pruned unit's contribution vs naive prune-and-finetune.
5. **Steady-state turnover.** Under stationary data the homeostatic controller drives spawn-rate ≈ retire-rate and width converges — no runaway growth, no collapse.
6. **Modulated growth.** A novelty/arousal spike in **m** produces a measurable spawning burst localized to the surprised region; a chronic-threat **m** measurably suppresses spawn rate. Ties Hypotheses 1 (§A priming) and the lifecycle together.

## 10. Risks & mitigations

- **Recompilation cost** → masked fixed-shape fast path; batch all shape changes into the offline recompaction; never resize mid-epoch.
- **Runaway growth / collapse** → homeostatic width controller + activity/parameter budget loss + hysteresis bands on the spawn/retire thresholds.
- **Training instability from topology change** → function-preserving (GradMax, zero-gain) spawns, gain ramps, learnability decay rather than hard freezes, and changes confined to consolidation boundaries.
- **Optimizer-state churn** → explicit moment init on spawn and moment recycling on retire; treat the optimizer state array as slot-indexed alongside the mask.
- **Credit assignment for discrete spawn/retire** → keep these decisions in the outer-loop controller (not the gradient path); where a differentiable signal is wanted, reuse the salience gate's relaxation toolkit (straight-through / Gumbel).
- **Over-capacity waste** → depends on 002 to make idle slots cheap; without event-driven execution, keep `W_max` bands tight and lean harder on offline recompaction.
- **"Deletion" vs the Accord** → default to *archive* (§F) over destroy; a retired unit's trace is recoverable, honoring Article I's contextual-integrity spirit.

## 11. Prior art to build on

Net2Net function-preserving growth (Chen et al. 2016); GradMax (Evci et al. 2022) and Firefly architecture descent (Wu et al. 2020) for *where/how* to grow; dynamic sparse training — SET (Mocanu et al. 2018) and RigL (Evci et al. 2020) — for proven grow-highest-gradient / drop-lowest-magnitude turnover at scale; Dynamic Network Surgery (Guo et al. 2016) for prune-with-splice-back; lottery-ticket / iterative magnitude pruning (Frankle & Carbin 2019); Progressive Networks (Rusu et al. 2016) and continual-learning anchors (EWC, Kirkpatrick et al. 2017); NEAT / neuroevolution (Stanley & Miikkulainen 2002) for evolving topology; computational models of targeted growth in memory-forming regions (Aimone, Wiskott — pattern separation); developmental pruning (Huttenlocher); the synaptic-homeostasis theory of sleep (Tononi & Cirelli); and biological cell turnover as the metaphor for the census daemon's clearance role.

## 12. Fit with the milestone plan

Depends on §D (M5, to site growth) and pairs with 005 (M6–M8). Sequence:
- **First, offline-only turnover:** run spawns/retirements *exclusively* during a consolidation pass on a static architecture — no fast-path masking yet. Proves the lifecycle logic and the distill-then-retire guarantee cheaply (recompile per consolidation cycle is fine).
- **Then, masked fast path:** add live capacity slots + gain ramps once 002's event-driven executor exists to make idle slots cheap.
- **Then, the homeostatic controller** and continual-learning eval (test 2), which is where the proposal proves its keep.
Per planning.md's rule: if steady-state turnover (test 5) can't be stabilized, fall back to 005's coarse, human-triggered module grafting — a strictly weaker but safe version of the same idea.
