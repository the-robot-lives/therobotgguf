# Avenue 5 — Mathematical instruments (the honest version of "magical breakthroughs")

**Channel:** brief item 4 ("magical breakthroughs in topology or other maths if
you're feeling up to it").
**Honesty note (Accord, Appendix B.2):** no breakthrough mathematics is on
offer here, and pretending otherwise would be the one way to fail this brief.
What the scan *did* surface is unusually good: several rigorous, active
mathematical programs whose objects map one-to-one onto components therobot has
already built — mostly as **instruments** (measure, certify, alarm) and twice
as **foundations** (the math semvec and the settling decoder were implicitly
assuming). Instruments are how new mathematics actually enters engineering;
the two clearly-labeled conjectures at the end are where the topology brief is
taken at its word.
**Evidence base:** survey §5.

---

## 5.1 Contraction-certified settling **[extend — flagged as an open gap]**

**What.** Proposal 004's settle loop currently relies on step caps and
empirical convergence. Monotone operator equilibrium networks (2006.08591),
Jacobian-regularized DEQs (2106.14342), and recurrent equilibrium networks
with built-in contraction certificates (2104.05942) supply the missing
theory: parameterize (or regularize) the settling map so a fixed point
provably exists, is unique per basin, and is reached at a known rate.

**The instrument.** The measured contraction rate is a *free confidence
signal*: fast contraction = the model is sure; slow/non-contraction = genuine
ambiguity (bistability, 004's own design signature) — exposed on the control
surface next to 4.8's unsettledness gauge. Nobody has bounded the contraction
constant of a settling loop on a frozen LLM donor.

**First experiment.** Estimate the local Lipschitz constant of the E7 loop
empirically (power iteration on perturbed canvases) across easy/hard/ambiguous
prompts; check rate-vs-difficulty correlation; add Jacobian regularization to
the graft-training recipe where it oscillates. **Effort:** analysis + a small
training-loss term.

## 5.2 Semvec as a formally-constructed relative-representation frame **[extend — the foundation item]**

**What.** Three published results jointly *predict* semvec should work, and
none of them built it: relative representations (2209.15430 — anchor-relative
coordinates are invariant across seeds/architectures, enabling zero-shot
stitching), the linear representation hypothesis with a causal inner product
(2311.03658 — which metric makes concept directions behave), and the Platonic
Representation Hypothesis + vec2vec (2405.07987, 2505.12540 — representational
convergence across models is real and exploitable even unsupervised).

**The upgrade.** Reconstruct semvec explicitly in this frame: (a) the labeled
corpus *is* an anchor set — analyze axis stability under anchor resampling
(the theory says which invariances — rescaling, permutation — to quotient
out); (b) fit the causal inner product on the donor and use it (not raw
cosine) for semvec_query similarity; (c) structure the 512 dims as a product
manifold — Euclidean factors for named ordinal axes, a hyperbolic factor
(1705.08039 lineage; mixture-of-curvature LLM evidence 2505.24722) for the
topic/taxonomy block, where hierarchy embeds with provably lower distortion;
(d) keep vec2vec as the unsupervised fallback for aligning a donor when
labels are scarce. Cross-donor portability (SV4) stops being an empirical hope
and becomes the theory's predicted behavior — with the anchor-stability
analysis as its certificate.

**First experiment.** Re-run the v1 admission table under 5 anchor
resamplings; report per-axis frame stability; fit the causal inner product on
0.8B recordings and re-score semvec_query's zero-shot suite against cosine.

## 5.3 Koopman spectroscopy of the state banks **[extend — flagged as an open gap]**

**What.** E4's banks are *designed* with timescales (fast/mid/slow/glacial,
learned α). Koopman operator methods (2407.06312 and lineage) measure the
timescales a dynamical system *actually exhibits*: fit a Koopman operator on
bank-trajectory recordings; its eigenvalue spectrum is the empirical leak
ledger.

**The instrument.** Design-vs-reality audit (did the glacial bank actually
learn to be glacial, or did training collapse it to fast?); drift detection
over long sessions (eigenvalues wandering toward 1 = the state interference
risk proposal 003 §9 flags, now measurable); and a principled way to choose
bank widths per timescale (spectral mass per band).

**First experiment.** Record bank trajectories over the H1 suite post
graft-training; EDMD fit; compare spectrum to the learned α's. A pure
analysis pass over data the tests already produce.

## 5.4 Singular-learning-theory gauges for the module economy **[extend]**

**What.** The local learning coefficient (2308.12108) is a
singularity-aware effective-complexity measure, computable by SGLD sampling;
refined per-component LLCs (2410.02984) track when individual heads
specialize; developmental-stage detection (2402.02364) finds phase
transitions in training.

**The instrument.** Three registry/census uses: (a) module admission gains a
complexity score — two shims with equal effect but different rLLC are not
equal (the simpler one composes more safely); (b) shim/graft training gets
phase-transition monitoring (the moment a steering module "groks" its
attribute is a stage boundary devinterp can detect); (c) proposal 006's
census gets a principled dormancy signal — a unit cohort whose restricted LLC
stops changing has consolidated, which is the *definition* of 006's
CONSOLIDATING→DORMANT transition, currently specced by heuristic vitality.

**First experiment.** rLLC traces while training one shim family; check the
trace's knee aligns with admission-score saturation.

## 5.5 Topology and intrinsic dimension as placement and health metrics **[extend]**

**What.** Intrinsic-dimension profiles across depth show a universal
expansion-compression shape with semantic content peaking at the ID trough
(2302.00294); persistent homology gives unsupervised embedding-quality and
collapse/fragmentation detection (1906.00722 lineage, recent PH quality
metrics).

**The instrument.** (a) **Tap placement**: extraction-v1's two-pass slice
search currently sweeps depths uniformly — compute the ID profile first and
concentrate the survey near the trough, where the semantics are; (b)
**lifecycle telemetry**: track PH/ID of a module's activation cloud across
its life (spawn → specialize → retire) — a fragmenting cloud is a
malfunctioning module, a collapsing one is a redundant module (006's
redundancy-retire signal, made topological); (c) **semvec health**: PH of the
projected semvec cloud per session as a monitor for representational collapse
under heavy steering (avenue-1.5's safety rail).

**First experiment.** ID-by-depth curve for the 0.8B from existing
recordings; overlay against the v0/v1 admission tables — do admitted sites
cluster at the trough? (If yes: cheaper surveys forever. If no: a finding
about *this* donor worth recording either way.)

## 5.6 Sheaf-consistency for the modulator bus **[extend — flagged as an open gap]**

**What.** A cellular sheaf (2012.06333, 2202.04579) attaches a vector space
to each node of a graph and linear restriction maps to edges; the sheaf
Laplacian measures how far local sections are from agreeing globally, with an
obstruction theory (cohomology) for when *no* global agreement exists.
therobot's bus — taps, modulator channels, shim gates, memory reads, each
with its own local view projected from the residual stream — is literally
this structure: sites are nodes, the learned projections are restriction
maps.

**The instrument.** The sheaf-Laplacian residual over the live readouts is a
**global workspace coherence alarm**: low residual = the model's sites agree
on what is being represented; a spike = internal contradiction (one site
reads menace, another reads calm) — which is exactly when 4.1 should ruminate
deeper, E5 should write, and a guardrail should look. Harmonic sections
(residual-zero states) characterize the bus's self-consistent "beliefs."

**First experiment.** Build the sheaf from the admitted-site projections;
compute residual traces over the behavioral suites; check spikes co-occur
with H1 late-flips and salience-gate fires. Analysis-only; no graph change.

## 5.7 Attractor and mean-field theory for the settle canvas **[adopt]**

**What.** Modern Hopfield theory (2008.02217) gives attention exponential
attractor capacity and one-step convergence bounds — the formal backing for
E5-as-attractor-memory (avenue-4.4's fixed-point seeds) and 004's lock-in
story. The mean-field transformer analysis (2312.10794) proves long-time
token *clustering* — which predicts what over-settling does to a canvas
(interpretations collapse into degenerate agreement) and hence supplies a
principled stop-time: halt settling before the clustering regime, not just
when change < ε.

**First experiment.** Measure canvas token-diversity across settle steps on
long generations; identify the clustering onset; set the step cap from it
rather than by hand.

## 5.8 Steering safety geometry: tropical boundaries and conceptor algebra **[extend]**

**What.** Tropical geometry (1805.07091) counts/locates the linear-region
boundaries of ReLU-family networks — a shim edit that crosses many boundaries
is a qualitatively bigger intervention than its vector norm suggests.
Conceptors (2410.16314) replace steering *vectors* with ellipsoidal *regions*
composable by boolean algebra (AND/OR/NOT of behaviors) — a richer, still
runtime-cheap shim payload with composition laws that are actual laws.

**The instrument + upgrade.** (a) Report boundary-crossing counts alongside
crosstalk in shim admission (a second, geometry-aware selectivity score); (b)
prototype conceptor-shims for compound behaviors ("formal AND non-menacing")
where vector addition currently fights itself — P4's Gram-Schmidt is the
linear shadow of what conceptor algebra does properly.

**First experiment.** For admitted steering vectors at increasing gain,
count activation-pattern changes (cheap proxy for region crossings) and
correlate with the teacher-judged off-target drift SV3 measures.

## 5.9 Two conjectures, clearly labeled **[novel — speculation, kept honest]**

Filed as conjectures because the brief asked for topology-flavored reach;
neither has evidence yet, both are falsifiable with therobot's instruments.

**C-1 (Shim operad).** Module composition (E8 stacks, depends/conflicts)
currently has engineering rules but no algebra. Conjecture: admitted shims
over disjoint or nested semvec axis-sets form an operad-like compositional
structure (categorical-DL framing, 2402.15332) — i.e., there exist composition
laws under which admission scores of composites are *predictable* (within
bounds) from components. If true: admission cost for composites collapses
from testing every pair to verifying the laws once. Test: measure
composite-vs-predicted selectivity across the registry's verified pairs; the
operad exists exactly where the prediction holds.

**C-2 (Homotopy of settling basins).** 004 predicts bistability (two
attractors, young/old woman). Conjecture: the basin structure of the settle
map over a fixed prompt family is itself a stable, probe-able object — its
persistent-homology signature (number/robustness of basins across
perturbation scales) is a *task-difficulty invariant* that predicts
steps-to-settle and error rates better than surface features do. Test: PH of
settled-state clouds under canvas-seed perturbation vs the difficulty
proportionality data 004's tests already collect. If the signature is
unstable run-to-run, the conjecture dies honestly.

---

## Sequencing note

Every item except 5.9 is analysis-first over recordings and traces the
pipeline already produces — they are cheap relative to the GPU roadmap and
several (5.2, 5.5) directly sharpen extraction-v1 work already queued. 5.1
and 5.7 belong with 004's build-out. 5.9 waits until the instruments it needs
(PH tooling from 5.5, registry composition data) exist as by-products.
