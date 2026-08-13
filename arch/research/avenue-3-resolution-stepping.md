# Avenue 3 — Resolution stepping: collapse, expansion, and nested capacity

**Channel:** brief item 2 ("stepping up and down resolution … a 3-dimensional
structure (somewhat like MoE routing) that can go from thick to thinner …
collapsing a big model down by grouping blocks of matrices … expanding a
smaller model so that prior to further training it has identical impact to the
original").
**Baseline it builds on:** proposal 005 (frozen-core accretion), proposal 006
(unit lifecycle / masked capacity slots), the R-pipeline, feature negotiation.
**Evidence base:** survey §3. Both halves of the brief exist with real
guarantees — exact function-preserving expansion (LEMON, HyperCloning, the
composable-op algebra) and calibrated grouping-collapse (SliceGPT rotations,
OT neuron merging, layer folding, SVD-averaged recursive blocks). What does not
exist is the round trip, the nested artifact with certificates, or any of it
done as a *retrofit* that leaves the parent bit-identical.

---

## 3.1 The collapse toolbox as an R-stage ("R9 collapse") **[adopt]**

**Thesis.** Wrap the four proven grouping operators as converter stages, each
emitting an admission record: (a) **rotate-then-slice** (SliceGPT 2401.15024 —
orthogonal rotations leave the function exactly unchanged; then cut
low-variance subspaces), (b) **neuron merging by optimal transport**
(DOTResize 2507.04517 — a soft many-to-one grouping matrix applied to weights,
beats deletion), (c) **layer-group collapse** (LaCo 2402.11187 delta-merge;
Model Folding 2502.10216 data-free channel clustering with statistics repair),
(d) **head grouping** (GQA uptraining 2305.13245 — mean-pool KV heads, the
deployed precedent). Calibration data = the recordings R1 already produces.

**Why it wins.** "Collapse by grouping blocks of matrices" stops being a
hypothesis and becomes a menu with measured costs. Every later item in this
avenue consumes these operators.

**First experiment.** 0.8B → three collapse settings (mild/moderate/aggressive)
per operator; report perplexity, eval suite, and — unlike the literature — the
full semvec admission table per setting (see 3.4). **Risk:** low; these are
reproductions with better instrumentation.

## 3.2 The nested-tier GGUF (one artifact, thick or thin) **[novel — flagged as an open gap]**

**Thesis.** MatFormer shipped in Gemma 3n and Nemotron Elastic (2511.16664)
proved one artifact can carry nested submodels — but they ship extracted
checkpoints, and both retrain the whole net. Define the GGUF extension
instead: **prefix-ordered tensors + per-tier metadata + per-tier parity/eval
certificates**, so one file mmaps at width/depth tier T with the loader
enforcing tier admission the way it enforces feature negotiation today.

**Mechanism.** `therobot.tiers`: each tier declares its slice widths / live
layer set, its admission scores, and its certificate (eval hashes). Runtime
loads tier T by mapping tensor prefixes — no duplicate weights. Collapse tiers
are produced by 3.1; the full-width tier is the untouched donor (bit-exact by
the superset invariant).

**Why it wins.** Self-host reality: one download, laptop runs tier-1, the
server runs tier-3, and the *same* module registry targets both (modules
declare their minimum tier). No published file format does this.

**First experiment.** Two-tier file (donor + one moderate collapse); loader
tier negotiation; verify tier-0 parity gate still reports max |logit diff| = 0.
**Risk:** tensor-layout surgery in the exporter; contained to R7.

## 3.3 Exact expansion ladder (small→big with an identity certificate) **[adopt/extend]**

**Thesis.** The brief's expansion ask is solved in the literature — the task is
to *operationalize with certificates*. LEMON (2310.07999) gives exact expansion
for Pre-LN transformers at arbitrary width factors; HyperCloning (2409.12903)
gives exact logit preservation via block tiling; Gesmundo & Maile (2308.06103)
give six composable exactly-preserving ops covering every axis. Implement
`robotgguf expand` emitting the expanded checkpoint *plus a parity
certificate* (the R8 harness re-used: max |logit diff| over the vocab, target
= 0), then train only the new capacity (LLaMA Pro pattern, 2401.02415 — which
is proposal 005's economics arriving from the other direction).

**One scale-caution to encode:** G_stack (2405.15319) found exact-identity
init can *slow* subsequent training (gradient-tied twins); the recipe is exact
at t=0, then deliberate symmetry breaking (noise/masks) — record the break as
part of the certificate.

**First experiment.** 0.8B → 2× width via LEMON; certificate must read 0;
brief continued training on new capacity only; H-suite regression on the
frozen paths must be bit-identical. **Risk:** low; math is published, harness
exists.

## 3.4 Probe-gated collapse admission (behavioral certificates per merge) **[novel — flagged as an open gap]**

**Thesis.** The compression literature validates with global perplexity;
LaCo's cosine check is the field's most granular gate. therobot can gate every
grouping operation the way it gates shims: after each block merge / slice /
fold, re-run the semvec admission table and require **per-axis decodability and
selectivity within ε of the parent** — a *semantic* certificate that the
collapsed tier still represents what the parent represented, axis by axis.
Dropped axes are recorded findings ("tier 2 loses fine register distinctions"),
which becomes honest tier labeling for the module registry.

**Why it wins.** It converts compression from "how much perplexity did we pay"
to "which capabilities did we keep" — and no published collapse operator is
validated this way. This is also what makes 3.2's certificates *mean*
something.

**First experiment.** Run 3.1's sweep with per-axis deltas; find the collapse
knee per axis; publish the map (which concepts die first under each operator —
a genuinely new empirical result). **Risk:** none; measurement.

## 3.5 Collapse into recursion: depth becomes a runtime dial **[extend]**

**Thesis.** Relaxed Recursive Transformers (2410.20672) collapse layer groups
into one shared block (SVD-averaged) + per-depth LoRA; Mixture-of-Recursions
(2507.10524) routes per-token recursion depth over such a block. Compose them
on a *frozen donor* as an overlay (avenue-2.1 machinery): the shared block +
LoRA set is grafted, gated, and parity-checked, giving a model whose effective
depth is a per-token routed decision — the "thick to thinner based on
requirements" of the brief, vertically.

**Why it wins.** Depth-recursion is the natural meeting point of this avenue
and E7's iterate-until-quiet executor: one loop controller serves settle
(iterate the canvas) and recursion (iterate the block). It also inherits
continuous depth-wise batching throughput wins (~2-3× reported).

**First experiment.** Collapse layers 8-16 of the 0.8B into one recursive
block per RRT recipe (recordings as distillation data); measure quality vs
recursion count; wire the modulator to the depth router (anxious → deeper) and
measure the H2-style dial. **Risk:** the largest training spend in this avenue
short of 3.7; sequence after 3.1/3.4 prove the instrumentation.

## 3.6 Dormant capacity as a shipping feature **[extend — flagged as an open gap]**

**Thesis.** Masked Structural Growth (2305.02869) uses masks only during
training ramp-up; proposal 006 designs masked capacity slots for training.
Ship them: export artifacts that carry **dormant, masked slots** (weights
present, gain = 0, function-preserving by construction) that a later
`robotgguf wake` can activate and train into — on-device capacity expansion
without re-downloading a bigger model, with parity-at-wake as the certificate.

**Why it wins.** It is 006's lifecycle made deployable: the file grows the way
the organism does. Combined with 3.2, a single artifact spans tiers *below*
the donor (collapsed) and *above* it (dormant headroom).

**First experiment.** Export 0.8B + 10% dormant width; verify dormant parity;
wake and brief-train on a narrow task; verify frozen-path bit-exactness (H6
machinery). **Risk:** file-size cost of dormant weights — mitigate by shipping
them as low-rank factors or quantized seeds.

## 3.7 Error-pressure-targeted growth **[extend — flagged as an open gap]**

**Thesis.** Every published growth schedule is global ("stack at step T").
Proposal 006 already specifies the missing local signal — spawn where
cleave-point aux loss or settling residual concentrates — and the scan
confirms nobody in the LLM literature does targeted growth (Firefly-style
splitting never made it past vision). Implement the 006 spawn rule using 3.3's
exact operators as the insertion mechanism, at cleave-point granularity first
(coarser than 006's per-unit dream, buildable now).

**First experiment.** Curriculum with a deliberately under-served regime;
grow at the highest-error cleave point vs uniform growth at equal params;
measure regime accuracy delta. **Risk:** the signal may be too noisy at small
scale; that is a recorded finding that shapes 006's homeostat design.

## 3.8 µP-consistent tier families **[adopt]**

**Thesis.** If tiers (3.2), expansions (3.3), and recursive collapses (3.5)
are to be *trained further* cheaply, hyperparameters must transfer across
sizes: µP (2203.03466) makes optimal LR/init width-invariant. Adopt µ-param
for all newly-created capacity (grafts, expansions, dormant slots) so one
tuned recipe governs every tier — the training-side companion without which a
resolution ladder becomes a hyperparameter tax.

**First experiment.** LR-transfer check: tune on the smallest tier, verify
optimum holds at 2× per µTransfer; document as converter policy. **Risk:**
µP interacts with the donor's non-µP pretraining — applies cleanly only to
*new* parameters, which is fine (they are the only ones training).

## 3.9 Thin-proposes / thick-verifies serving **[adopt/extend]**

**Thesis.** LayerSkip (2404.16710) showed a nested model can draft with its
thin tier and verify with its thick tier — same weights, shared KV, exact
speculative acceptance. With 3.2's nested tiers this comes almost free:
tier-1 drafts, tier-3 verifies, and the E6 delta executor's "quiet blocks"
logic decides *when* the thick verify is even needed. Nested consistency (the
tiers were built by grouping, so they agree by construction where the
representation survived 3.4's gates) is exactly what speculative decoding
wants.

**First experiment.** Two-tier self-speculation on the 0.8B pair from 3.2;
acceptance rate and wall-clock vs dense. **Risk:** low; established
technique, new substrate.

---

## Sequencing note

3.1 → 3.4 (instrumented collapse) is the foundation and is mostly measurement.
3.2 and 3.3 are converter/format work that consume it. 3.6 rides 3.3's
machinery; 3.9 rides 3.2. 3.5 and 3.7 are the training-heavy items — sequence
behind the extraction-v1 GPU work already queued. 3.8 is policy, not project.
