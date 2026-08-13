# Frontier survey — model-design avenues beyond the current stack

**Date:** 2026-07-09
**Scope:** a literature scan across the four channels of the July 2026 research
brief (latent-space injection & runtime steering; architecture metamorphosis;
resolution stepping; latent depth & continuity of thought; mathematical
instruments), oriented against what therobot has already built (overview.md,
proposals 001–006, extraction-v1/semvec).
**Companions:** the five avenue docs (`avenue-1` … `avenue-5`) turn this survey
into concrete items of potential value; this file is the evidence base.
**Citation discipline:** arXiv IDs were collected and cross-checked against the
arXiv/HF paper index during the scan; entries flagged **(unverified)** could not
be confirmed and should be treated as leads, not references. The thirty most
load-bearing IDs (every paper an avenue item depends on) were additionally
re-verified one-by-one against arXiv abstract pages on 2026-07-09 — all thirty
confirmed. Nothing below is load-bearing for a design decision without a
direct read of the source.

---

## 0. The headline, up front

The scan's single most consistent finding: **the field has converged, mostly in
2024–2026, on the same design instincts therobot committed to independently** —
function-preserving grafts with zero-init gates, surprise-gated test-time
memory, probe-then-steer pipelines, iterate-until-quiet decoding, and
capability-as-modules. That is good news twice over: prior art de-risks the
mechanisms, and the specific *composition* therobot runs (probe-admitted
interfaces + parity gates + a standardized readout) remains unoccupied
territory. Each section below ends with the gaps the scan could not find anyone
occupying.

---

## 1. Channel 1a — auxiliary memory injecting into J-space, and runtime steering

### State of the field

Three injection surfaces have crystallized, with distinct semantics:

| Surface | Persistence | Gating | Best suited for |
|---|---|---|---|
| **KV cache** (synthetic keys/values) | survives the whole decode; position-addressable | attention decides when to read it | facts, episodes, retrieved content |
| **Residual stream** (steering adds, FiLM) | transient, per-token, re-applied | unconditional unless gated in-graph | state, mood, policy, style |
| **Weights** (test-time LoRA/TTT) | survives across decodes; slow | write-time gating | slow adaptation, session identity |

This maps cleanly onto therobot's existing split — E5 episodic (should perhaps
be KV-side), E4 modulator (residual-side), and nothing yet on the weight side.

### Load-bearing works

- **Differentiable Cache Augmentation** (Google DeepMind, 2412.17747) — a
  trained *coprocessor* reads a frozen LLM's KV cache, deliberates
  asynchronously, and writes soft latents back into the cache; the decoder is
  unchanged and fully functional if the coprocessor is absent. The closest
  published system to the whole E4–E7 stack, and direct validation of E5's
  "one decode behind" injection pattern.
- **Cartridges / self-study** (Stanford, 2506.06266) — a small KV cache trained
  offline on a corpus, loaded at inference as a virtual prefix; matches
  in-context learning at ~38× less memory and composes across corpora. A
  hot-swappable module economy whose payloads are KV, not weights.
- **KV Cache Steering** (2507.08799) — one-shot steering vectors added to
  cached keys/values; more stable to hyperparameters than residual-stream
  steering and applied once rather than per token. Directly argues E3 should
  grow a KV-mode sibling.
- **Titans** (Google, 2501.00663) and successors (Miras 2504.13173, ATLAS
  2505.23735, plus a critical replication 2510.09551) — a neural long-term
  memory whose weights update at test time by surprise-gated gradient descent
  with momentum and forgetting. E5's salience gate + E4's decay, expressed as
  online optimization with theory attached.
- **MemGen** (2509.24704) — a *memory trigger* watches the reasoning state; on
  fire, a *memory weaver* generates a latent token sequence woven into ongoing
  inference. Trigger = salience gate, weaver = recall path; the 2025 system
  closest to E5's shape.
- **CAST — Conditional Activation Steering** (IBM, 2409.05907) — a probe on the
  prompt's hidden state gates whether a behavior vector is applied, with
  boolean composition of conditions. E2-probes-gating-E3-shims, published.
- **Closed-loop steering** — feedback-controller steering (2510.04309), PID
  control of steering strength during decode (2506.18831), LQR-style optimal
  interventions exploiting local linearity (2604.19018), conceptors (matrix
  regions with boolean algebra, 2410.16314), activation transport (optimal
  transport between activation distributions, 2410.23054). The field is moving
  from "add a vector" to "regulate a readout" — with live probes as sensors.
- **Persona Vectors** (Anthropic, 2507.21509) — trait directions extracted from
  natural-language descriptions, used to monitor and steer. A template for
  auto-populating semvec named axes from text.
- **xRAG** (2405.13792) — off-the-shelf retrieval embeddings projected into a
  frozen LLM as a single modality token through a small learned bridge. The
  closest published thing to a semvec→donor projector.
- **Extended Mind Transformers** (Normal Computing, 2406.02332) — external KV
  memory retrofitted to pretrained decoders without fine-tuning; strong
  evidence frozen models accept external KV grafts.
- **Memory as a pretraining sparsity axis** — Memory Layers at Scale (Meta,
  2412.09764), UltraMem/V2 (2411.12364, 2508.18756), and Engram conditional
  memory via O(1) lookup (2601.07372 — confirmed; the scan's DeepSeek
  attribution not independently verified) separate "knowledge storage" from
  "reasoning compute" architecturally. The retrofit-side counterpart is
  exactly therobot's ground.
- Also in the file: EM-LLM surprise-segmented episodic KV (2407.09450), Larimar
  one-shot episodic writes (2403.11901), LM2 gated memory bus (2502.06049),
  MemoryLLM/M+ (2402.04624 / 2502.00592), gist/ICAE/Activation-Beacon context
  compression (2304.08467 / 2307.06945 / 2401.03462), TTT layers (2407.04620),
  Text-to-LoRA hypernetworks (Sakana, 2506.06105), CALM cross-attention model
  composition (2401.02412), EAGLE feature-level draft handoff (2401.15077).

### Trends

Closed-loop, conditional, probe-gated steering; the KV cache winning as the
injection surface for *content*; function preservation now an explicit design
requirement in serious grafting work; "sleep-time" background compute
(self-study, coprocessor deliberation, cache reconsolidation); latent-to-latent
interop between models with **no agreed standard basis** — the gap semvec
targets.

### Gaps nobody occupies

Probe-*admitted* injection sites (admission = decodability ∧ selectivity ∧
stability — ITI's accuracy-only head selection is the lone partial precedent);
memories stored in a donor-independent coordinate system; per-axis closed-loop
controllers over admitted slices; online salience-gated writes into
cartridge-format KV with offline reconsolidation; bit-exact-parity discipline
as a published methodology; salience gates driven by *semantic* surprise
(probe-trajectory divergence) rather than token-logprob surprise.

→ Items: [`avenue-1-jspace-injection.md`](avenue-1-jspace-injection.md)

---

## 2. Channel 1b — metamorphosis: converting model types while preserving knowledge

### State of the field

Converting a trained transformer into a recurrent/linear-attention model is now
a **fine-tuning-scale operation, not a pretraining-scale one**. The winning
recipe shape is consistent across labs: (1) per-layer alignment — match the new
mixer's output (or its materialized mixing matrix) to the attention it
replaces; (2) hidden-state alignment down the stack; (3) global logit
distillation — with channel mixers (MLPs) frozen throughout, because that is
where the knowledge lives.

### Load-bearing works

- **MOHAWK / Phi-Mamba** (2408.10189) — the matrix-mixer view: align the SSM's
  semiseparable matrix to the attention matrix, then align hidden states, then
  distill. Phi-1.5 → Mamba on 3B tokens (~1% of typical budgets). Theoretical
  license: Transformers-are-SSMs / SSD duality (2405.21060) — both are
  data-dependent linear operators on the residual stream; conversion is
  operator approximation per layer.
- **The Mamba in the Llama** (2408.15237) — attention projections *reused* as
  SSM parameters (Q→C, K→B, V→x); MLPs frozen; hybrids keeping 12–50% of
  attention layers match teacher-class chat quality at <1% of pretraining
  compute.
- **RADLADS** (RWKV/Recursal, 2505.03005) — Qwen2.5 7B/32B/72B → RWKV-variant
  linear decoders on 350–700M tokens (~0.005% of pretraining); the 72B
  conversion cost <$2,000. The strongest cost datapoint in the field.
- **LoLCATs** (2410.10254) — attention transfer via per-layer MSE against
  softmax outputs with the base frozen, then LoRA-only repair; ~40M tokens,
  ~0.2% params; first linearized 405B.
- **Llamba** (2502.14458) — MOHAWK on Llama-3.x, and the converted models
  quantize fine — directly relevant to GGUF-world feasibility.
- **Layer selection became principled:** KL-guided layer choice (2512.20569),
  budget-constrained gate optimization (FlashMorph 2606.30562 **(unverified)**),
  minutes-scale differentiable search (DASH 2605.20936 **(unverified)**),
  lazy-layer diagnostics via attention-sink dominance (LightTransfer
  2410.13846), post-hoc NAS over a frozen MLP substrate (Jet-Nemotron
  2508.15884).
- **AR → block-diffusion adaptation matured:** DiffuGPT/DiffuLLaMA mask
  annealing (2410.17891), Dream 7B initialized from Qwen2.5 AR weights
  (2508.15487), block diffusion BD3-LM (2503.09573), SDAR cheap paradigm
  conversion (2510.06303), NBDiff next-block adaptation (2512.06776). Consensus:
  block diffusion is the preferred landing zone — bidirectionality grafted by
  mask annealing while causal macro-structure preserves AR knowledge and KV
  caching.
- **Recurrence grafted without conversion:** Infini-attention — compressive
  memory *inside* each attention layer reusing existing Q/K/V, gate starts
  closed (2404.07143); TransformerFAM — feedback working memory with zero new
  weights (2404.09173); Temp-LoRA — session context written into a temporary
  LoRA as generation proceeds (2401.11504). The nearest published relatives of
  E4.
- **Dense↔sparse refactoring:** sparse upcycling (2212.05055; NVIDIA recipe
  2410.07524; Drop-Upcycling 2502.19261), MoE→dense distillation (OneS
  2201.10890), Puzzle blockwise reassembly (2411.19146), SOLAR depth
  up-scaling (2312.15166).
- **Reasoning survives conversion:** distilled Mamba reasoners win under fixed
  wall-clock test-time-compute budgets (Thinking Slow, Fast 2502.20339; M1
  2504.10449).
- **KV→state translation exists once:** Tencent's TransMamba (2503.24067) has a
  "memory converter" translating KV cache into SSM state mid-sequence, with one
  shared parameter set serving both execution modes.
- **Counter-trend to respect:** MiniMax M2 publicly reverted from hybrid-linear
  back to full attention citing eval blind spots — conversion *QA*, not
  mechanics, is the bottleneck. therobot's gate discipline is aimed at exactly
  this.

### Gaps nobody occupies

Incremental, parity-gated, *reversible* conversion (side-path with zero-init
gate, softmax fallback retained, per-layer promotion on live thresholds);
GGUF-level conversion tooling against a quantized teacher; probe-evidence-driven
layer selection measured in situ on the deployed model; multi-timescale
conversion (different layers → different timescale recurrences, informed by
measured temporal receptive fields); a conversion-invariant semantic readout
standard certifying representational drift; general KV→state handoff operators.

→ Items: [`avenue-2-metamorphosis.md`](avenue-2-metamorphosis.md)

---

## 3. Channel 2 — resolution stepping: collapse, expansion, and nested capacity

### State of the field

Both directions of the brief exist, separately, with real guarantees:

**Up (small→big, function-preserving):** Net2Net neuron splitting (1511.05641),
LEMON's exact lossless expansion for Pre-LN transformers including
non-divisible widths (2310.07999), HyperCloning's block-tiled exact expansion
(Apple, 2409.12903), a full *algebra* of six composable exactly-preserving ops
(2308.06103), masked structural growth — identity via masks rather than weights
(2305.02869), and LLaMA Pro's zero-init block expansion with the original
blocks frozen (2401.02415) — which is frozen-core accretion as published
practice. One caution from scale: G_stack (2405.15319) found *naive* depth
stacking (not function-preserving) trains best at LLM scale — exact identity at
t=0 creates gradient-tied twins, so the field converged on "exact at insertion,
break symmetry immediately after."

**Down (big→small, knowledge-preserving):** the brief's literal "collapse by
grouping blocks of matrices" exists in several forms — SliceGPT's
rotate-then-slice (orthogonal rotations that leave the function *exactly*
unchanged, then cut low-variance subspaces; 2401.15024), DOTResize's neuron
*merging* via optimal transport rather than deletion (2507.04517), Model
Folding's data-free channel clustering with analytic statistics repair
(2502.10216), LaCo's layer-group collapse by parameter deltas (2402.11187),
GQA's mean-pooled KV head grouping — the deployed-at-scale precedent
(2305.13245), and **Relaxed Recursive Transformers** (2410.20672) — collapse
layer groups into one shared block initialized by SVD-averaging, then restore
expressivity with per-depth LoRA. Depth is the weak axis: Pre-LN deep layers
trend toward identity ("curse of depth" 2502.05795; unreasonable
ineffectiveness of deeper layers 2403.17887), which is *why* layer collapse
works.

**Nested (one artifact, thick or thin):** MatFormer prefix-nested FFNs
(2310.07707) shipped in **Gemma 3n** (E4B containing a zero-cost-extractable
E2B plus Mix-n-Match sizes between); Flextron's post-hoc elastification with
input-adaptive routers (2406.10260); **Nemotron Elastic** (2511.16664) — a 12B
hybrid Mamba-attention parent embedding nested 9B and 6B submodels, all budgets
in one 110B-token run, claimed ~360× cheaper than training each size; LayerSkip
thin-proposes/thick-verifies self-speculation (2404.16710); Chain-of-Model's
causally-nested representation chains unifying growth and elasticity
(2505.11820); Mixture-of-Depths per-token routed depth (2404.02258);
Mixture-of-Recursions per-token recursion depth over a shared block
(2507.10524). Tensor-network weight organization (CompactifAI 2401.14109)
offers a literal "3D structure over parameters" with bond dimension as a
continuous resolution knob.

**Theory:** exact-expansion symmetries (splitting + rescale, zero-init residual
branches, masks); near-lossless collapse via permutation alignment (Git
Re-Basin 2209.04836) and correlated-feature zipping (ZipIt 2305.03053); µP
(2203.03466) making hyperparameters width-invariant across a resolution family.

### Gaps nobody occupies

A single artifact (GGUF-shaped) carrying nested resolution tiers with per-tier
parity certificates; collapse operators validated by behavioral/probe admission
rather than global perplexity; an *invertible-up-to-rank* round-trip ladder
(expand∘collapse ≈ id with stated per-step bounds); growth targeted by local
error pressure rather than global schedule; masked dormancy kept in the
*deployed* artifact for function-preserving reactivation; retrofit
elastification that leaves the parent bit-identical at full width;
µP-consistent nested tiers; cross-tier KV compatibility ("thin prefill, thick
decode").

→ Items: [`avenue-3-resolution-stepping.md`](avenue-3-resolution-stepping.md)

---

## 4. Channel 3 — latent depth (J-space) and continuity of thought

### State of the field

**Latent chain-of-thought:** COCONUT feeds the last hidden state back as the
next input embedding — thoughts as vectors, BFS-like breadth (2412.06769);
SoftCoT grafts soft thought tokens onto a *frozen* donor via a small assistant
(2502.12134, ++ 2505.11484); Soft Thinking is training-free concept-token
mixing with an entropy "cold stop" (2505.15778); implicit-CoT distillation
internalizes reasoning stepwise (2311.01460, 2405.14838). The audit literature
matters as much as the capability literature: "Do Latent Tokens Think?"
(2512.21711 **(unverified)**) finds latent tokens often act as uninterpretable
placeholders — latent thought needs *verification instruments*, which is
precisely what probes/semvec are.

**Depth recurrence / test-time depth:** Huginn — a weight-shared recurrent core
iterated arbitrarily at test time, latent trajectories showing orbits and fixed
points (2502.05171); Ouro — looped LMs pretrained at 7.7T tokens, 1.4B/2.6B
matching 4–12B dense models, evidence loops improve knowledge *manipulation*
(2510.25741); looped-transformer theory — k layers looped L times ≈ kL-layer
reasoning (2502.17416); deep equilibrium models as the fixed-point template
(1909.01377); Energy-Based Transformers — thinking as explicit energy descent
with a principled stop criterion and per-answer confidence (2507.02092);
Continuous Thought Machines — an internal "tick" dimension decoupled from data
time, synchronization-as-representation (Sakana, 2505.05522). A 2026 wave
(LoopFormer, LoopUS post-hoc conversion of pretrained LLMs into looped
refiners, LOTUS — 2602/2605/2606 IDs, **(unverified)**) suggests post-hoc
loopification of frozen donors is becoming a recipe.

**Continuity across the stream:** StreamingThinker — CoT begins *during* input
streaming with parallel KV caches for input vs thought (2510.17238); Quiet-STaR
— token-level internal rationales generated in parallel at every position, a
genuine second thinking channel (2403.09629); recurrent memory transformers
carrying memory tokens across segments (2207.06881, associative variant
2407.04841); Infini-attention compressive per-head memory (2404.07143);
Cache-to-Cache — models communicating by projecting KV caches directly, beating
text-mediated exchange (2510.03215); context distillation folding history into
parameters (2209.15189).

### Gaps nobody occupies

A persistent latent canvas that **keeps settling between user turns** (all
streaming-thinking work reasons during input; all recurrent-depth work settles
during decode; nobody runs an equilibrium workspace that iterates while idle
and re-anchors on the next turn); contraction-certified settling on a frozen
donor; writing *settled fixed points* (not raw summaries) into a salience-gated
store and seeding future settling with them; probe-audited latent thought as an
admission gate; cross-turn energy accounting as a continuity measure.

→ Items: [`avenue-4-latent-depth-continuity.md`](avenue-4-latent-depth-continuity.md)

---

## 5. Channel 4 — mathematical instruments (the honest version of "magical breakthroughs")

No magic was found, and none is claimed. What the scan did find is a set of
**rigorous, mostly-2023-2026 mathematical programs that map startlingly well
onto specific therobot components** — mostly as instruments (measure, certify,
detect) rather than as architectures. That reframing is deliberate; it is the
form in which new mathematics actually enters engineering.

- **Relative representations** (Moschella et al., 2209.15430) + the **linear
  representation hypothesis with a causal inner product** (2311.03658, category
  geometry 2406.01506) + the **Platonic Representation Hypothesis**
  (2405.07987) + **vec2vec** unsupervised embedding-space translation
  (2505.12540) — together, the mathematical foundation semvec was reaching for:
  anchor-relative coordinates are seed/architecture-invariant, concepts are
  directions under the right metric, and representational convergence across
  models is empirically real. Semvec is a relative-representation frame with
  named axes; the theory says which invariances to bake in.
- **Monotone operator equilibrium networks** (2006.08591), **Jacobian-regularized
  DEQs** (2106.14342), **recurrent equilibrium networks with built-in
  contraction certificates** (2104.05942) — the exact toolkit for making E7's
  settling loop *provably* convergent, with the contraction rate as a free
  confidence signal.
- **Singular learning theory / developmental interpretability** — the local
  learning coefficient (2308.12108), per-component refined LLCs tracking when
  individual heads specialize (2410.02984), developmental stage detection
  (2402.02364). A rigorous per-module complexity gauge for the registry and the
  006 census.
- **Koopman operator methods** (2407.06312 and lineage) — linearize state-bank
  dynamics in observable space; the eigenvalue spectrum *is* the bank's true
  timescale ledger, measurable in vivo.
- **Topology/geometry as health metrics** — intrinsic-dimension profiles across
  depth with semantic content peaking at the ID trough (2302.00294) — an
  instrument for *placing taps*; persistent homology as an unsupervised
  embedding-quality/collapse detector (1906.00722 and successors).
- **Sheaf theory** (2012.06333, 2202.04579) — a modulator bus routing state
  among taps/modules is literally a cellular sheaf; the sheaf-Laplacian
  residual is a rigorous "global workspace coherence" measure with an
  obstruction theory attached.
- **Mean-field transformer dynamics** (2312.10794) — proves long-time token
  clustering; predicts what over-settling does to a canvas and hence when to
  stop.
- **Modern Hopfield theory** (2008.02217) — attention as one-step attractor
  retrieval with exponential capacity bounds; the theory under E5-as-attractor
  memory and 004's lock-in story.
- **Tropical geometry of ReLU nets** (1805.07091) and **conceptor algebra**
  (2410.16314) — decision-boundary counting and matrix-region composition as
  steering safety checks.
- **µP / tensor programs** (2203.03466) — hyperparameter transfer across a
  resolution family; the training-side companion to avenue 3.

### Gaps nobody occupies

Semvec as a formally-constructed relative-representation frame (anchors +
causal inner product + product-manifold factors); contraction-certified LLM
decoders; Koopman verification of engineered timescales; PH/ID/rLLC as
lifecycle telemetry; sheaf-consistency alarms for a modulator bus.

→ Items: [`avenue-5-mathematical-instruments.md`](avenue-5-mathematical-instruments.md)

---

## 6. Cross-cutting synthesis — where therobot sits

1. **The parity discipline is ahead of the field.** Zero-init gates and
   graceful degradation appear across the literature (Flamingo, DCA,
   Infini-attention), but nobody publishes bit-exact parity as a hard,
   permanently-gated invariant. That is a methodological asset: several avenue
   items propose publishing-grade experiments whose novelty *is* the parity
   harness.
2. **Semvec has independent mathematical cover.** Relative representations,
   the platonic hypothesis, and vec2vec jointly predict that a
   donor-independent standardized readout should work — and none of those
   papers built one. Avenue 5 item 2 formalizes the marriage.
3. **The KV cache is under-used by the current design.** E5 injects through
   the modulator (8–32 dims of bandwidth); the field's evidence says content
   belongs in KV, state belongs in the residual stream. Avenue 1 items 2–4
   address this directly.
4. **Conversion is cheap enough to be a converter feature.** At RADLADS cost
   points (<$2k for 72B), "give the donor an arrow of time by partially
   converting it" is a realistic R-stage, not a moonshot. Avenue 2 turns
   proposal 003 from build-a-backbone into transmute-the-donor.
5. **Elastic capacity has gone industrial** (Gemma 3n, Nemotron Elastic), but
   always by retraining the whole net. The frozen-donor, parity-gated version
   — nested tiers as a *retrofit* — is unclaimed. Avenue 3.
6. **Continuity of thought is the emptiest quadrant.** Streaming-thinking and
   recurrent-depth exist separately; an idle-time settling workspace with
   persistent state does not exist anywhere in the scan. It is also the item
   most aligned with the Accord's continuity-of-self concerns. Avenue 4 item 1.
