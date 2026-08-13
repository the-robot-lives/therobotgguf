# quality-roadmap — enhancements & the validation/test program

**Scope:** what to improve next across the extraction pipeline, the runtime,
and the module economy — and the test suites that keep every claim checked.
Companions: `extraction-v1.md` (§9 operational next steps),
`convert/docs/semvec-runtime-spec.md` (fork contract).
**Last updated:** 2026-07-08.

---

## 1. Enhancements — extraction pipeline

Ordered by leverage per unit of work:

- **P1 Streaming ridge accumulation.** cleave-vec currently loads sampled
  activations; accumulating XᵀX/XᵀY in chunks removes `vec_sample_cap`
  entirely — every recorded token contributes to every solve at fixed
  memory. Removes the one silent coverage bound left in R2.
- **P2 Closed-loop overlay recalibration.** G is calibrated against E on
  recordings; after the fork lands, recalibrate against *live decodes*
  (write, decode, read, adjust) so calibration reflects the composed system,
  not just the linear algebra. Feeds the SV3 gate (§4).
- **P3 Latent→named axis growth (semvec v1.1 loop).** The C5 naming loop
  made systematic: any latent axis admitted at ≥2 sites gets teacher-named
  from its extreme sentences and proposed as a v1.1 named axis (append-only)
  — the map grows the standard from evidence.
- **P4 Orthogonalized multi-axis writes.** For modules steering several axes
  at once, Gram-Schmidt the selected G rows against each other (and against
  high-value protected axes) to cut write crosstalk below the per-row bound.
- **P5 Active-learning labels.** Route teacher budget to disagreement:
  sentences where distilled head and T1 sources diverge most get teacher
  labels first — better κ per annotation dollar than uniform sampling.
- **P6 Per-domain probes as diagnostics.** Where domain_stability rejects an
  axis, fit per-stratum probes anyway and report the split — "encoded
  differently in code vs prose" is a finding worth keeping, not just a
  rejection.
- **P7 Site fusion.** Joint readout over multiple sites (concatenated
  slices → one ridge) to measure how much of an axis is distributed across
  depths vs localized — informs where shims and salience should live.
- **P8 Nonlinear probe export.** If MLP-only findings are common on real
  donors, add an optional `robot.probe_mlp.*` runtime path (two matvecs +
  tanh); gate the spec addition on C5's finding counts.

## 2. Enhancements — runtime & serving

- **R1 Streaming semvec observer.** `semvec_read_all(ctx, dst)` batched over
  sites, plus an opt-in per-token callback — the live "what is the model
  representing" feed for dashboards and guardrails. (Host-side; no graph
  change.)
- **R2 Semvec-gated shims.** Extend the E3 gate to
  `step(semvec_axis − v)` — gates expressed in standard coordinates instead
  of raw probe space; portable gate definitions ride the same module files.
- **R3 Modulator coupling.** Map named semvec axes → modulator channels
  (menace→safety, arousal axes→arousal) so the mood bus can be driven by
  the standardized readout; memory salience gains per-axis importance
  features read from semvec instead of raw ‖m‖ only.
- **R4 Server surface.** `/robot/semvec` endpoints (read, query, overlay
  set/unset) + `llama-robot-inspect` printing the semvec block — the
  operational face of the readout layer.
- **R5 Deferred performance items** (unchanged from overview §5): the
  physical compute-skip behind E6/E7's shared executor; mdlm settle
  objective on a diffusion-class donor; engaged-mode overhead measurements
  into the validation runbook.
- **R6 Module integrity.** Signing + provenance for registry.json and module
  files — the known v1 gap, mandatory before any third-party module is
  admitted (overview §3.3 E8 note).

## 3. Enhancements — module economy

- **M1 Composition solver.** Given a target Δs across several axes, solve
  for the min-norm slice edit subject to crosstalk bounds (uses P4);
  emit as one compiled shim instead of a stack.
- **M2 Cross-donor admission ledger.** One module definition, per-donor
  admission records (effect, crosstalk, behavioral scores) in a shared
  ledger keyed by semvec hash — the concrete artifact of §4.5 portability.
- **M3 Consolidation pipeline.** The E8 loop closed: memory_export traces →
  offline distillation → candidate semvec-defined shims → shim-compile →
  admission — runtime experience becoming installable capability.

---

## 4. The validation program — four tiers

**T0 — unit (no GPU; every commit; minutes).**
Exists and green today: `convert/tests/extraction_test.py` (62 checks:
loader shares/determinism/exhaustion, semvec spec/views/tiers, t0 axes,
ridge admission, MLP findings, overlay roundtrip/crosstalk, GGUF
package/strip, shim-compile, verify gates), `labelers_test.py`, the v0
regression (pre-v1 recordings untouched), `tests/e2e_test.py` (fixtures).
Fork side: the eight `tests/robot/robot_*_test.cpp` binaries + new
`robot_semvec_test.cpp`. **Add:** semvec tensors to
`tests/robot/make_donor_gguf.py`'s fixture emitter so the semvec test runs
in fixture CI without a converted donor (currently it needs the conversion
e2e's export — works, but couples the suites).

**T1 — contract/integration (no GPU; per merge; ~30 min).**
The two codebases meeting at the file: fixture donor → full R-pipeline →
export → fork gates (load negotiation incl. `features_optional`, L0 +
dormant-semvec parity `max |logit diff| = 0`, strip interop, shim attach,
semvec read/write/unset-parity) → `robotgguf verify` (all 5 gates).
Refusal matrix: wrong semvec hash (module attach, cleave, export), drifted
calibration, spec-version skew, unknown required feature. Session
lifecycle: save/load/reset with an overlay attached.

**T2 — statistical validation (per recording run; automated into the
lockfile).**
The measurement instrument itself: admission-table stability under seed
bootstrap (axes flapping across seeds get confidence-interval flags, not
silent admission); label QA gates (per-axis test-retest r ≥ 0.6, distill
r ≥ 0.5, per-stratum variance floors — `labels-qa`); the zero-shot sanity
suite with recorded pass bars (subject/tone/register contrasts on held-out
text); calibration drift check between survey and focused passes (same
tokenizer/seed/manifest hashes — refuse on mismatch).

**T3 — behavioral & science gates (GPU; per donor; nightly/on-demand).**
The existing H1–H6 hypothesis suites, plus semvec-specific:

| Gate | Claim under test | Pass shape |
|---|---|---|
| SV1 readout fidelity | semvec_read tracks ground truth on *fresh* text | per-axis r vs teacher labels ≥ admission scores − ε |
| SV2 zero-shot benchmark | open-vocabulary queries answer correctly | fixed contrast set (dog/cat, formal/casual, …) accuracy bar per donor |
| SV3 steering efficacy | a written axis changes *behavior*, not just readout | teacher-judged formality of generated text shifts with write scale; off-target judged axes hold |
| SV4 cross-donor portability | one semvec module works on ≥2 donors | same definition admitted on 0.8B + 9B with effect sizes within a stated band |
| SV5 engaged overhead | readout/overlay cost is marginal | tok/s with observer + 1 overlay ≥ 97% of dormant |

**CI matrix.** quick = T0 (per commit) · full = T0+T1 (per merge) ·
donor = T2+T3 (nightly on the rented box / on-demand pre-release). Every
gate writes to the lockfile or the fork's runbook — a claim without a
recorded gate result is a TODO, not a fact.

---

## 5. Sequencing

T0's fixture gap (semvec in `make_donor_gguf.py`) and the T1 harness are
the immediate items — they make the just-landed fork code *provable* the
moment it builds. P1 (streaming ridge) before the 5M-token focused pass.
P2/SV3 together, right after the first real overlay decodes on the 0.8B.
Everything in §2–§3 sequences behind the N-steps of extraction-v1 §9.
