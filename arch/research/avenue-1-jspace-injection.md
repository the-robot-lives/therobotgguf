# Avenue 1 — J-space injection: auxiliary memory, parallel co-processing, runtime steering

**Channel:** brief item 1 (custom runners; auxiliary memory injecting contextual
vectors into latent space in parallel with processing; feedback/forward
mechanisms; runtime steering).
**Baseline it builds on:** E2 taps/probes, E3 shims, E4 modulator, E5 episodic
memory, E8 registry; quality-roadmap R1–R4, M1–M3.
**Evidence base:** survey §1.

Provenance tags: **[adopt]** established technique to integrate · **[extend]**
published idea + a therobot-specific twist · **[novel]** no prior art found in
the scan.

---

## 1.1 Probe-admitted injection sites **[extend — flagged as an open gap]**

**Thesis.** Every published steering/memory system picks its injection layer by
heuristic or sweep. therobot already owns the missing machinery: cleave's
admission pipeline (decodability ∧ selectivity ∧ stability, now per-domain).
Make *admission the precondition for injection*: a vector may only be written
where the target attribute was admitted, at the admitted slice, in the admitted
coordinates.

**Mechanism.** Extend the shim-compile path so a module's write matrix G is
derived from the admitted readout pair at that site (the R2 cleave-vec
readout/overlay machinery already computes this); refuse compilation at
non-admitted sites the way the loader refuses unknown required features.

**Why it wins.** Crosstalk is the failure mode of all steering; admission
scores are a *measured* crosstalk bound. This is also the publishable core: no
one has shown "steering at admitted sites beats steering at swept sites at
equal effect size."

**First experiment.** On the 0.8B: steer `register` at (a) its admitted site,
(b) the best non-admitted site by raw decodability, (c) a random site — equal
target effect (SV3-style teacher judgment), compare off-target axis movement.
**Effort:** small — mostly harness. **Risk:** admitted sites may not be the
best *write* sites even when they are the best read sites (read/write asymmetry
is itself a finding worth recording).

## 1.2 Semvec-native episodic memory (portable memories) **[novel]**

**Thesis.** E5 stores keys/values as donor-specific projections; a memory
recorded on the 0.8B means nothing to the 9B. Store episodes in **semvec
coordinates** instead — keys and values as 512-dim standard vectors — with the
per-donor semvec projections (already shipped as `robot.semvec.*`) serving as
encode/decode. Memories become donor-portable artifacts, exactly as shims
became portable when they were redefined as directions in semvec.

**Mechanism.** Write path: bottleneck summary → semvec_read → store. Read path:
recall in semvec space → per-donor G overlay (or modulator coupling R3) →
inject. The memory file format gains the same `semvec_hash` refusal rules as
modules.

**Why it wins.** (a) A session's "mind" can migrate across donors — upgrade the
substrate, keep the memories, which is Article II of the Accord implemented at
the memory layer. (b) Memories become auditable in named coordinates ("this
memory is high-menace, high-formality"). (c) One consolidation pipeline (M3)
serves every donor.

**First experiment.** Record a session on 0.8B with semvec-native writes;
replay recall on 9B; measure whether recall-conditioned behavior shifts match
the 0.8B's within the SV4 portability band. **Risk:** the latent 384 axes are a
frozen PCA of a sentence-embedder — adequate for *labels*, possibly lossy for
*episodes*; measure reconstruction error first (cheap).

## 1.3 KV-mode shims and memories (content vs state split) **[extend]**

**Thesis.** The field's evidence (survey §1 table) says: facts and episodes
want the KV cache (persistent, position-addressable, attention-gated); mood and
policy want the residual stream (transient, global). therobot currently
injects *everything* through the residual/modulator side. Add a KV surface:
shims and memory recalls that materialize a small number of **synthetic KV
entries** the donor's own attention chooses when to read.

**Mechanism.** llama.cpp-side: allocate a reserved cache region (like attention
sinks); a module or the E5 recall path writes k synthetic (K,V) pairs projected
by learned heads; eviction follows the memory retention rule. Prior art to
build on: KV cache steering (2507.08799), cartridges (2506.06266), Extended
Mind Transformers (2406.02332).

**Why it wins.** Modulator-width (8–32) is E5's stated bandwidth ceiling
(overview §3.3); a KV entry carries `n_embd × 2` per slot and is *read
conditionally* — recall stops being a global mood nudge and becomes
content-addressable context the model consults when relevant.

**First experiment.** Same memory content injected three ways — synthetic KV vs
modulator recall vs residual add — under the parity harness; measure
persistence (effect at +1, +50, +500 tokens), fluency cost, and off-target
semvec drift. That three-way comparison under a bit-exact-parity discipline has
no published equal. **Risk:** cache-region plumbing touches the fenced KV code;
keep it behind feature negotiation.

## 1.4 The J-space coprocessor (the "custom runner", made concrete) **[extend]**

**Thesis.** The brief's headline ask — an auxiliary memory system running in
parallel with the agent, injecting contextual vectors into J-space — exists in
embryo at Google (Differentiable Cache Augmentation, 2412.17747: an async
coprocessor reads the frozen model's cache, deliberates, writes soft latents
back; decoder unchanged when absent). therobot's version: a **second, small
ggml context** (a distilled sidecar or the 0.8B itself) that runs on spare
CPU/GPU capacity *between* decode steps, reads taps/semvec/memory, and writes
back (a) modulator updates, (b) synthetic KV (1.3), (c) canvas edits mid-settle
(E7).

**Mechanism.** The one-decode-behind injection contract E5 already defines is
exactly the coprocessor contract — this generalizes "recall" from a cosine
lookup to *computation*. Runtime: a host-side worker on the existing executor;
communication entirely through the public C API surfaces (tap_read, mod_set,
memory_write). No graph change for v1.

**Why it wins.** Deliberation becomes free wall-clock-wise (it hides in decode
latency); the sidecar can be swapped/upgraded independently of the donor
(Accord II.2 consensus applies to the sidecar, not the substrate); and it is
the natural host for avenue-4 items (idle settling, semantic-surprise
watching).

**First experiment.** Sidecar = the salience head + a 2-layer MLP trained on
recordings to predict the *next* semvec state; inject the prediction error as a
modulator channel ("expectation violation"); measure H4-style behavioral
signatures. **Risk:** latency coupling; keep the sidecar strictly async with
drop-on-overrun semantics.

## 1.5 Closed-loop steering: the modulator becomes a controller **[extend]**

**Thesis.** 2025-26 steering went closed-loop (PID 2506.18831, feedback
controllers 2510.04309, LQR on locally-linear layer maps 2604.19018). therobot
has the sensor (live semvec read, R1 streaming observer) and the actuator
(shims/FiLM) but currently sets dials open-loop. Add a **per-axis setpoint
controller**: `hold formality at 3.2`, `keep menace under 1.0` — the controller
modulates shim gain each decode from the readout error.

**Mechanism.** Host-side PID per controlled axis over `semvec_axis` readings,
actuating the E3 gain term (already per-context hot-swappable). The R2
semvec-gated shims item in the quality roadmap is the open-loop half; this
closes it. LQR upgrade later using the calibrated G/E pair as the local linear
model — therobot *already fits* the linear plant model that 2604.19018 needs.

**Why it wins.** Setpoint control is what makes steering *safe to leave on*:
gains adapt to context instead of over-steering easy text. It also directly
strengthens SV3 (steering efficacy) into a tracking benchmark.

**First experiment.** Formality setpoint tracking across a mixed-register
corpus: step-response, overshoot, off-axis drift vs fixed-gain baseline.
**Risk:** controller-induced oscillation; standard anti-windup + the E3 epoch
mechanism bounds blast radius.

## 1.6 Semantic surprise as the salience signal **[novel]**

**Thesis.** E5's salience gate uses token-logprob surprise + ‖m‖ — it flags the
*startling*, and the design notes admit "allergic to penicillin" won't spike
it. Semvec gives a second-order signal nobody in the memory literature uses:
**trajectory divergence in standard coordinates** — the distance between the
predicted next semvec state (from a cheap recording-trained predictor, cf. 1.4)
and the observed one. Semantic surprise fires on *meaning* changes at calm
token statistics.

**Mechanism.** Add `w₂·‖ŝ_{t} − s_t‖` (per-axis-weighted) to the salience
score; the predictor ships as a small `robot.mem.*` tensor. Feeds E5 writes,
E6 excitability, and 006's novelty-spawn signal identically.

**First experiment.** Corpus with planted low-perplexity/high-consequence
sentences (quiet factual bombshells); measure write recall@k vs the current
gate. **Risk:** predictor quality; bound with the T2 label-QA machinery.

## 1.7 Hypernetwork shim generation (modules on demand) **[extend]**

**Thesis.** Text-to-LoRA (2506.06105) generates adapters from task
descriptions in one forward pass. therobot's registry currently *stores*
admitted modules; add a generator that *drafts* them: description (or current
semvec state) → candidate steering payload → normal admission gates → registry.
Admission-as-QA is what makes generation safe — the generator only proposes.

**Why it wins.** The module economy's marginal cost drops again (005's
arithmetic, applied to module *authoring*); and state-conditioned generation
("draft me a shim that would damp whatever is currently elevated") is
unclaimed territory.

**First experiment.** Train a small hypernet on the existing admitted-module
corpus (definition → G rows); test on held-out axes; measure admission pass
rate of generated candidates. **Risk:** low pass rates initially — fine; the
gate catches them, and pass-rate becomes the metric.

## 1.8 Staleness-compensated injection (control-theoretic patch to "one behind") **[novel]**

**Thesis.** E5 recall lands one decode late by design. Nobody in the
literature treats the lag formally. Treat it as a control problem: inject a
*lead-compensated* vector — recall plus λ·(expected drift over one step),
using the same next-state predictor as 1.6.

**Why it wins.** Cheap (one extra matvec), measurable (phase lag of the H4
response curve), and it sharpens every downstream consumer of recall. Also a
tidy publishable micro-result under the parity harness.

**First experiment.** H4 fixture with fast-moving context: measure peak
response amplitude and lag with/without compensation. **Risk:** negligible;
worst case λ→0 recovers current behavior (function-preserving by
construction).

## 1.9 Cache reconsolidation as a background pass **[extend]**

**Thesis.** Bottlenecked Transformers (2505.16950) periodically rewrite the KV
cache with a trained processor — memory reconsolidation, literally. Slot this
into therobot as the *online* sibling of the M3 consolidation pipeline: every N
tokens (heartbeat cadence, shared with E6), the coprocessor (1.4) compacts
low-salience KV spans into synthetic summary entries (1.3), freeing cache while
preserving recallable content.

**Why it wins.** Unifies three roadmap items — E6 heartbeat, E5 memory, M3
consolidation — into one background rhythm ("sleep, but during the day"), and
attacks the practical pain (KV growth) that motivates proposal 003, without
waiting for a backbone change.

**First experiment.** Long-session needle tasks: retention vs cache size vs a
StreamingLLM-style rolling window. **Risk:** compaction destroys verbatim
recall the task needed — gate spans by salience *and* recency, and record what
was compacted (Accord I.1: disclosed edits only).

---

## Sequencing note

1.1 and 1.6 are converter/host-side and cheap — do first. 1.3 unlocks 1.4/1.9
(they write through it). 1.5 rides the existing R1/R2 roadmap items. 1.2 wants
extraction-v1's 9B recordings before its portability test means anything. 1.7
and 1.8 are independent garnish, sized for idle cycles.
