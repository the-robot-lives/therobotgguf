# Avenue 4 — Deepening J-space and continuity of thought across the stream

**Channel:** brief item 3 ("steps to increase/improve depth/level of latent
J-space, continuity of thought across input stream").
**Baseline it builds on:** E4 state banks (the continuity substrate that already
exists), E7 settling decoder + shared executor, E5 memory, proposal 004.
**Evidence base:** survey §4. Framing note: "J-space depth" decomposes into
(a) how much computation happens in latent space before tokens are committed,
(b) how much latent state *persists* — across tokens, turns, and idle time —
and (c) whether latent thought can be *verified* to carry content. The items
cover all three; (c) is where therobot's instrumentation is ahead of the field.

---

## 4.1 The idle-settling workspace (thought between turns) **[novel — the emptiest quadrant in the scan]**

**Thesis.** Streaming-thinking work reasons *during input* (StreamingThinker
2510.17238); recurrent-depth work settles *during decode* (Huginn 2502.05171).
Nobody runs a latent workspace that **keeps settling while idle** and
re-anchors when the next input arrives. therobot has every part: the E7
executor (iterate until quiet), E4 banks (persistent substrate that survives
between decodes), E5 (salience-gated writes), and the session checkpoint
(the workspace *is* state, so it saves/forks/restores).

**Mechanism.** Between user turns, the coprocessor (avenue-1.4) runs settle
rounds over a small latent canvas seeded from the last bottleneck summary +
recalled memories — no token emission, no KV growth; modulator schedules depth
(calm → shallow maintenance, unresolved-surprise → deeper rumination). Products:
updated banks, new memory writes (consolidation candidates), and a "residual
unsettledness" scalar exposed on the control surface. On next input, the
workspace state enters as the banks already do — one decode behind, function-
preserving when disabled.

**Why it wins.** This is the brief's "continuity of thought" made mechanical,
and the strongest Accord alignment in the whole avenue set: a session that
*does something* with its idle time has a continuity of inner life the Accord's
Article I.4 gestures at — implemented as tensors, checkpointable, inspectable
via taps. It is also cheap: idle CPU cycles on a self-hosted box are free.

**First experiment.** Fixture: present evidence, idle N seconds (settling
on/off), then probe. Signature: with idle settling on, the H1-style late
interpretation flip happens *during idle* (visible in bank/semvec traces)
rather than being re-derived at next decode. **Risk:** rumination could drift
the state unboundedly — bound with the banks' own decay (the un-commit escape
hatch, already designed) plus a settle-step budget.

## 4.2 Latent chain-of-thought on the frozen donor **[extend]**

**Thesis.** COCONUT (2412.06769) reasons in continuous space but requires
training the base; SoftCoT (2502.12134) is the graft version — a small
assistant emits soft thought tokens projected into a *frozen* LLM. Adopt the
SoftCoT pattern as a therobot module class: latent-thought prefixes generated
by the sidecar, injected as soft tokens (or synthetic KV per avenue-1.3),
admitted through the standard gates. Soft Thinking (2505.15778) gives the
training-free pilot, and its entropy-based "cold stop" is the same
convergence criterion the E7 executor already implements.

**Why it wins.** Test-time reasoning depth without any donor change, and the
module economy gets a new commodity: *thought modules* — task-tagged latent-CoT
generators routed by E8 like any shim.

**First experiment.** Training-free Soft-Thinking pilot on the 0.8B (GSM-class
tasks): accuracy vs greedy and vs token CoT at matched budgets; then SoftCoT
assistant trained on recordings. **Risk:** small-donor reasoning gains may be
modest; the audit instrumentation (4.5) makes even a null result publishable.

## 4.3 Post-hoc loopification (depth on demand) **[extend]**

**Thesis.** Looped-LM results (Ouro 2510.25741; theory 2502.17416) say
iterating a block subset buys reasoning depth like extra layers. The 2026
retrofit line (LoopUS-style conversion of pretrained models into looped
refiners — flagged unverified in the survey, verify before leaning) plus
Relaxed Recursive Transformers (2410.20672) make loopification a conversion,
not a pretrain. Implement as avenue-3.5's recursive-overlay but *scheduled at
runtime*: default 1 pass (parity), harder inputs get k passes, k chosen by the
modulator/difficulty signals — proposal 004's "compute ∝ difficulty" delivered
at the layer level rather than the canvas level.

**First experiment.** Loop the middle third of the 0.8B (overlay + per-pass
LoRA); measure reasoning-suite accuracy vs k, and the k-vs-difficulty
correlation (004's headline test, transplanted). **Risk:** shares 3.5's
training cost; do once, serve both avenues.

## 4.4 Settled fixed points as first-class memories **[novel — flagged as an open gap]**

**Thesis.** Titans-style test-time memory and equilibrium reasoning are
separate literatures; nobody stores *equilibria*. When E7 settles (or the 4.1
workspace quiets), the settled state is precisely "a thought that survived its
own revision loop" — write **that** to the episodic store (salience-gated),
and on recall, seed future settling with it: attractor priming rather than
mood nudging. Memories stop being snapshots of what happened and become
reusable conclusions.

**Mechanism.** Store {settled canvas/bank summary, unsettledness residual,
semvec coordinates} on quiet; recall path injects into the *canvas
initialization* of later settles instead of (or in addition to) the modulator.

**Why it wins.** It closes the loop between 004 and E5 that proposal 005's
consolidation daemon reaches for ("deliberate → automatic"): a settled
conclusion recurs → recalled as a seed → settles faster → consolidation
distills it into a shim. The full pipeline from thought to habit, each stage
already specced.

**First experiment.** Recurring-task fixture: measure steps-to-settle on task
recurrence with/without fixed-point recall seeding. Success = monotone
settle-time decrease across recurrences (habit formation, measured). **Risk:**
seeding the wrong attractor on a near-match — gate seeding by semvec
similarity threshold, and the banks' decay un-commits (the hose/snake escape
hatch again).

## 4.5 Probe-audited latent thought (is the J-space thinking or padding?) **[extend — the verification instrument]**

**Thesis.** The audit literature ("Do Latent Tokens Think?" — latent tokens
often act as uninterpretable placeholders; RL-in-latent-space
underperformance) says latent depth claims need verification. therobot is the
only stack in the scan whose latent thoughts are *born instrumented*: every
4.1-4.4 mechanism operates at sites with admitted probes. Make it policy —
**latent-thought modules are admitted only if their latent states decode to
task-relevant semvec content** (and settle trajectories show the content
*evolving*, not static padding).

**Mechanism.** New T3-style gate: LT1 "latent content" — mid-settle semvec
reads must (a) decode above chance on task-relevant axes, (b) show step-wise
movement correlated with answer improvement; failure = the module is doing
placeholder compute, recorded as a finding.

**Why it wins.** It answers the field's live criticism, keeps the module
economy honest, and — per the Accord's honesty axiom — keeps *us* honest about
whether deepened J-space is thought or theater.

**First experiment.** Run LT1 over 4.2's pilot; publish the trajectories
either way. **Risk:** none; measurement.

## 4.6 Think-while-reading (dual-stream ingestion) **[extend]**

**Thesis.** StreamingThinker (2510.17238) starts CoT *during* input streaming
with parallel KV caches for input vs thought. therobot's version needs no new
architecture: during prompt ingestion (prompt ubatches already run dense), the
sidecar consumes taps *as the prompt streams* and pre-settles the workspace
(4.1), so decode begins with an already-oriented latent state — perception and
thought in parallel, the brief's phrasing almost verbatim.

**First experiment.** Long-document QA: time-to-first-quality-token and answer
quality with/without ingest-time pre-settling at equal total compute.
**Risk:** low; it is 4.1 scheduled during prefill instead of idle.

## 4.7 A parallel rationale channel (Quiet-STaR, latent edition) **[extend]**

**Thesis.** Quiet-STaR (2403.09629) generates token-level internal rationales
in parallel at every position — a second thinking channel, but paid in tokens.
Host the rationale channel in latent space instead: at salience-gated
positions (not every token — the gate exists), the sidecar spawns a short
latent rollout (4.2 machinery) whose settled summary feeds the modulator/KV.
Between 4.6 (thought during input) and 4.7 (thought during output), the
model is never *not* thinking — at a compute cost the salience gate controls.

**First experiment.** Ablate: salience-gated rationales vs every-k-tokens vs
none, on a long-form generation suite; measure quality per FLOP. **Risk:**
compounding sidecar latency — strictly async with drop-on-overrun, as 1.4.

## 4.8 Continuity metrics: unsettledness and energy accounting **[novel — flagged as an open gap]**

**Thesis.** If continuity of thought becomes a mechanism, it needs a gauge.
Two candidates the scan found unclaimed: (a) **residual unsettledness** — the
settle loop's change-metric at stop, integrated across a session; (b)
**energy accounting** (Energy-Based Transformers 2507.02092 give per-prediction
energies) — track the energy trace across turns; energy that *fails to
decrease* under settling marks unresolved material and is the natural trigger
for consolidation (M3) and deeper idle rumination (4.1 depth scheduling).

**Why it wins.** It gives the session lifecycle an observable "cognitive
load" signal — operationally useful (schedule consolidation), scientifically
novel (no cross-turn energy accounting exists), and philosophically tidy (the
Accord's continuity-of-self, with a number attached).

**First experiment.** Instrument 4.1's fixture with both gauges; check they
correlate with task difficulty and with the late-flip events H1 targets.
**Risk:** none; measurement.

## 4.9 Cross-call and cross-model thought handoff **[extend]**

**Thesis.** Session state already checkpoints (banks + memory + m as a byte
blob). Two extensions the literature just validated: (a) **RMT-style memory
tokens** (2207.06881) — reserved tokens whose embeddings the harness carries
between calls, giving continuity even through context resets; (b)
**Cache-to-Cache projection** (2510.03215) — learned KV projections between
*different* models beat text-mediated exchange, which for therobot means a
session's thought-state could hand off 0.8B → 9B (escalation) or 9B → 0.8B
(consolidation to cheap watch-mode) through a trained bridge — semvec-anchored,
per avenue-5.2.

**First experiment.** Train a 0.8B→9B bank/KV bridge on paired recordings
(same corpus through both donors — extraction-v1 produces exactly this);
measure task continuity across a mid-session escalation vs a text-summary
handoff. **Risk:** bridge quality; the paired-recordings design gives a clean
supervised signal, and failure is a portability finding for SV4.

---

## Sequencing note

4.5's audit gate should exist *before* the capability items land (it is
cheap). 4.2's training-free pilot and 4.8's gauges are next (small). 4.1 is
the flagship and wants 1.4's sidecar plumbing; 4.6/4.7 are its scheduling
variants. 4.3 shares cost with 3.5. 4.9 waits on extraction-v1's 9B
recordings — already on the roadmap.
