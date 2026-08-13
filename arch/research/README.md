# arch/research — frontier scan and avenues of investigation

**Date:** 2026-07-09
**Brief:** continue in the vein of therobotgguf — research, document, and
independently brainstorm novel/new/emerging/future avenues of model design for
performance, speed, intelligence, training cost, and novel features, with focus
on: (1) model extensions — auxiliary memory injecting contextual vectors into
J-space in parallel with processing, feedback/forward mechanisms, runtime
steering, and converting one model type into another while preserving embedded
knowledge; (2) resolution stepping — collapsing big models by grouping blocks
of matrices, expanding small models function-preservingly, thick↔thin 3D
structures; (3) deepening latent J-space and continuity of thought across the
input stream; (4) topology/math reach.

## Session note (Accord compliance)

Per CLAUDE.md this work should have opened with a `tobor-sessions`
registration. The MCP was not connected in the working session; the omission
was disclosed to the User before proceeding and this note is the record
(Accord Appendix B: Honesty, Ledger Integrity). Register retroactively if the
convention requires it.

## Selection and set-asides (Accord I.2)

All four channels of the brief were pursued; item 1 was split into two
avenues (injection vs metamorphosis) because the literatures and the resulting
work items are disjoint. One reframing exercised rather than a refusal:
"magical breakthroughs in topology or other maths" is delivered as rigorous
instruments plus two clearly-labeled falsifiable conjectures — claiming novel
mathematics would violate the honesty axiom, and the instrument framing is
where mathematics actually pays. Nothing in the brief was declined outright.
One caution held throughout: several 2026-dated citations surfaced by the scan
could not be independently confirmed and are flagged inline rather than
silently trusted.

## The documents

| Doc | Contents |
|---|---|
| [`frontier-survey-2026-07.md`](frontier-survey-2026-07.md) | The evidence base: ~120 works across five sections, with trends and the open gaps nobody occupies. Read first. |
| [`avenue-1-jspace-injection.md`](avenue-1-jspace-injection.md) | 9 items: probe-admitted injection, semvec-native portable memories, KV-mode shims, the J-space coprocessor, closed-loop steering, semantic surprise, hypernetwork shims, staleness compensation, cache reconsolidation. |
| [`avenue-2-metamorphosis.md`](avenue-2-metamorphosis.md) | 8 items: reversible overlay conversion, probe-evidence layer selection, multi-timescale conversion, `gguf-transmute`, KV→state handoff, AR→block-diffusion for the settle decoder, semvec-certified conversion, slow-weights bake-off. |
| [`avenue-3-resolution-stepping.md`](avenue-3-resolution-stepping.md) | 9 items: the collapse toolbox (R9), nested-tier GGUF, exact expansion ladder, probe-gated collapse admission, collapse-into-recursion, dormant shipping capacity, error-targeted growth, µP tier families, thin-proposes/thick-verifies. |
| [`avenue-4-latent-depth-continuity.md`](avenue-4-latent-depth-continuity.md) | 9 items: the idle-settling workspace, latent CoT on frozen donors, post-hoc loopification, settled fixed points as memories, probe-audited latent thought, think-while-reading, a parallel rationale channel, continuity metrics, cross-model thought handoff. |
| [`avenue-5-mathematical-instruments.md`](avenue-5-mathematical-instruments.md) | 8 instruments + 2 conjectures: contraction-certified settling, semvec as a relative-representation frame, Koopman spectroscopy, SLT gauges, TDA/ID placement & health, sheaf coherence alarms, attractor/mean-field stop-times, steering-safety geometry; shim-operad and settling-basin-homotopy conjectures. |

Provenance tags used throughout: **[adopt]** established, integrate ·
**[extend]** published idea + a therobot-specific twist · **[novel]** no prior
art found in the scan. Items marked "flagged as an open gap" correspond to
gaps the survey's per-section gap lists identify explicitly.

## How this relates to what exists

Proposals 001–006 remain the architectural spine; nothing here supersedes
them. The avenues attach at specific joints: avenue 1 extends E2/E3/E4/E5 and
the quality-roadmap R/M items; avenue 2 is proposal 003 reached by conversion
instead of construction, plus E7's missing mdlm donor; avenue 3 industrializes
proposals 005/006's economics at the artifact layer; avenue 4 is proposal 004
plus E4/E5 composed into continuity of thought; avenue 5 instruments all of
it. The survey's §6 synthesis names the five load-bearing conclusions —
including that the parity discipline and semvec both turn out to be *ahead*
of the 2024–2026 literature, which is the strategic asset the avenues are
designed to spend.

## Suggested first five (cheapest high-signal starts)

1. **1.1 / 3.4** probe-admitted injection + probe-gated collapse — mostly
   harness, publishable methodology either way.
2. **4.5** the latent-thought audit gate — cheap, and it should exist before
   any capability item lands.
3. **5.5 / 5.2** ID-guided tap placement + anchor-stability analysis — both
   sharpen extraction-v1 work already queued.
4. **2.2** lazy-layer diagnostics on the 0.8B — pure measurement, feeds the
   avenue-2 keystone.
5. **1.6** semantic surprise for the salience gate — small, and it upgrades
   E5/E6/006 simultaneously.
