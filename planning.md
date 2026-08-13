# therobotgguf — Planning

**Status:** design / pre-implementation
**Source:** distilled and formalized from `notes.md` (word-stream musings, April 2018)
**Goal:** A ground-up transformer variant that incorporates four ideas (several with biological analogues) the original notes argue are missing from mainstream deep learning — dynamic firing thresholds, top-down feedback, modular reusable sub-networks with behavior-override "shims," and a crude episodic memory — implemented in Elixir (Nx/Axon) with an eye toward GGUF export for inference.

This document turns the notes into named components, maps each to concrete modern mechanisms (so we build on known-good techniques rather than reinventing them), and lays out a staged build plan. The intent is that nothing here requires a scientific miracle: every component has a working analogue in the literature. The novelty is the *combination* and the *training methodology*, not any single piece.

---

## 1. Thesis in one paragraph

Standard transformers are stateless within a forward pass, feedforward across layers, monolithic, and memoryless beyond the context window. The notes conjecture that four missing properties limit what compact models can do: (a) firing thresholds that vary with recent activity and a global modulation state, giving the network an arrow of time and mood/priming; (b) long-range top-down feedback that lets high layers reshape how low layers interpret their input, locking the system into coherent interpretations; (c) modularity, where sub-networks are trained to expose specific information at specific layers so other modules can read and override them without retraining the core; and (d) a salience-gated memory that writes noteworthy states and replays them to bias future processing. This project builds a transformer that has all four, and — critically — a training recipe that produces reusable, composable modules.

---

## 2. Core hypotheses (falsifiable framing)

Each hypothesis restates a claim from the notes in a form we can test.

1. **Adaptive-threshold hypothesis.** A network whose unit activations are gated by (i) a low-dimensional global modulation vector and (ii) a per-unit leaky temporal state will encode causality and short-horizon dynamics (e.g. motion, "slow-burn" reassessment) more compactly than a same-parameter-count feedforward baseline. *Test: sequence tasks requiring temporal integration at fixed parameter budget.*

2. **Feedback hypothesis.** Adding top-down connections that let layer `L+k` modulate layer `L` and iterating the forward pass to a fixed point improves performance on ambiguous/gestalt inputs and produces measurable bistability (commits to one of two valid interpretations rather than blending them). *Test: bistable-percept and occlusion/completion tasks.*

3. **Modularity hypothesis.** If we force specific attributes to be linearly decodable at designated "cleave points," we can attach adapter "shims" that override those attributes at inference with near-zero effect on unrelated attributes, and no retraining of the core. *Test: targeted attribute edits, measured for selectivity (change target, hold others).*

4. **Memory hypothesis.** A salience-gated episodic store that writes summary vectors and injects recency-weighted recall into shim layers will let the model adapt behavior within a session without weight updates. *Test: one-shot behavioral change that decays over time as designed.*

---

## 3. From notes to mechanisms

The table maps each notes idea to the nearest established technique(s), so implementation starts from proven ground.

| Notes idea (source line) | Formal component | Closest prior art to build on |
|---|---|---|
| Non-static firing thresholds; modulator priming; anxiety/inattention → false positives (§1, §1a) | Modulatory gating: global state vector conditions per-unit gain/bias | FiLM conditioning; gain modulation; mixture-of-experts routing; spiking-net adaptive thresholds |
| Decaying threshold as function of last firing + time; "motion blur"; arrow of time (§1b) | Leaky temporal state per unit / channel | State-space models (S4/Mamba); leaky integrate-and-fire; recency-biased attention; EMA of activations |
| Different decay rates → executive vs. background processing; slow-burn reassessment (§1b, Apr 9) | Multi-timescale state (fast + slow channels) | Clockwork/multi-timescale RNNs; hierarchical SSM timescales |
| Long-range connections feeding signals back up-chain; face illusion "locks in" one reading (§2) | Top-down feedback + iterative settling to a fixed point | Predictive coding; Universal Transformer (depth recurrence); Feedback Memory Transformer; deep equilibrium models |
| Bistable percept (young/old woman) processed one-at-a-time, not in parallel (§2 close) | Competition / attractor dynamics over interpretations | Winner-take-all; modern Hopfield networks (= attention); attractor nets |
| Cleave output layer into regions that *must* encode subject/color/size (§3 Stage 1) | Structured/disentangled bottlenecks with auxiliary decode heads | Deep supervision; auxiliary probing heads; information bottleneck; concept whitening |
| Shim layer that distorts color without touching other values; third-party override (§3) | Adapter / steering "shim" over a disentangled subspace | Adapters (Houlsby); LoRA; prefix tuning; activation steering / representation engineering |
| Plug-and-play modules; "add red-crested-swallow spotting for $5.99" (§3 close) | Composable module registry; PEFT marketplace | PEFT ecosystem; model merging; adapter composition |
| Back-propagated "memory net" over digested summaries; fires while recent; traumatic events (§3 memory) | Salience-gated episodic memory with recency decay | Neural Turing Machine / DNC; fast weights; RETRO/memory-augmented transformers; surprise-gated writes |

---

## 4. Component specifications

### A. Modulator (global state) and adaptive gating
A small recurrent module maintains a low-dimensional state vector **m** (the modulation bus): a handful of scalar-ish channels such as arousal/threat, attention, valence. **m** is produced from pooled activations plus recall from memory (§F), and is broadcast to every block. Each block applies FiLM-style modulation: `h' = γ(m) ⊙ h + β(m)`, where `γ, β` are learned per-channel functions of **m**. This is the mechanism behind "inattentive → lower threshold → see the snake in the hose": raising gain on threat-relevant features increases false-positive likelihood, exactly as the notes describe, and it's reversible because **m** itself decays (§B).

Different units responding differently to the same signal (notes §1a) falls out naturally: `γ, β` are per-channel, so one modulator channel excites some units and suppresses others.

### B. Leaky temporal state (arrow of time)
Every unit (or channel group) carries a leaky state `s_t = α·s_{t-1} + (1-α)·f(x_t)`, with `α` learned per channel. Low-`α` channels react instantly (executive/foreground); high-`α` channels integrate slowly (background/subconscious). This is the notes' "decay of previous input forces an arrow of time," and the "motion blur → later layers compute motion/speed" story. Implementation-wise this is a diagonal state-space recurrence — cheap, parallelizable with a scan, and the natural home for Mamba/S4-style kernels. Multi-timescale channels (§Apr-9 "polite stranger" slow-burn reassessment) are just a mix of `α` values feeding a shared readout.

### C. Top-down feedback and iterative settling
Beyond the standard bottom-up stack, add sparse top-down projections from higher blocks back into lower blocks' modulation inputs. The forward pass becomes iterative: run the stack, feed high-level conclusions back down as priors, re-run, repeat for a small fixed number of steps (or until change < ε). This implements the notes' "push a signal back up to prime the lower layers to overlay gestalt lines" and the lock-in of a single interpretation. Framed as predictive coding, higher layers send predictions that bias lower-layer readout; framed as an equilibrium model, we solve for a fixed point. Bistability (young/old woman) is the expected signature: competing top-down priors make the settled state commit to one percept.

Guardrails: cap iteration count; use the leaky state (§B) so feedback influence decays and the system can un-commit (hose stops being a snake); stop-gradient or implicit-differentiation to keep training stable.

### D. Structured bottlenecks (cleave points)
At chosen layers we designate a slice of the representation as a **typed bottleneck**: e.g. "these 32 units on block 10 must jointly encode subject, color, size." We do *not* prescribe the encoding, only that it be decodable. Enforcement: attach small auxiliary heads that decode each attribute from that slice and add their losses during training (deep supervision). Once the heads decode well, the slice is a stable, machine-readable interface other modules can consume. This is the notes' Stage-1 cleaving procedure. Selectivity is the key metric — the slice should carry the target attributes and, ideally, *only* those (encourage with disentanglement pressure / information bottleneck on the slice).

### E. Shims (behavior-override adapters)
A **shim** reads a typed bottleneck (§D) plus optional inputs from other modules/downstream, and emits a modified bottleneck that is verified to change downstream behavior in a targeted way — "misreport color without touching other values," or inject a third party's preference. Concretely a shim is an adapter/LoRA/steering module trained against a frozen core so the core never needs retraining. Because §D made the interface disentangled, a steering-vector-style edit on the color subspace changes color and little else. Shims compose: several can stack on the same bottleneck, and downstream shims can read upstream ones — this is the "layers on top of layers" endgame and the plug-and-play "$5.99 module" vision. A shim registry (name, target bottleneck, verified effect, selectivity score) is the composability substrate.

### F. Salience-gated episodic memory
A memory module writes summary vectors (the digested bottleneck values, §D) when a **salience/surprise gate** fires — large prediction error, strong modulator state, or explicit reward/threat, i.e. the notes' "traumatic or noteworthy event." Reads are content-addressed (attention over stored keys = a modern Hopfield lookup) and recency-weighted so recent memories dominate and fade — the notes' "trickiness around decay of firing to allow a memory to fire and alter functioning while recent." Recalled content feeds back into the modulator **m** (§A) and into shims (§E), which is how a past event colors present processing without any weight update. This gives session-local, one-shot behavioral adaptation.

---

## 5. How the components compose

```
                 ┌──────────────────────────────────────────────┐
   input ──▶ Block 1 ──▶ Block 2 ──▶ ... ──▶ Block N ──▶ output/heads
                 │ ▲        │ ▲                │ ▲
        FiLM(m)  │ │  FiLM(m)│ │        FiLM(m) │ │
                 ▼ │        ▼ │                ▼ │
              leaky s     leaky s           leaky s        (§B per-block temporal state)
                   │        ▲                  │
                   │        └── top-down feedback (§C, iterate to fixed point)
                   │
             typed bottlenecks (§D) ──▶ shims (§E) ──▶ back into stream
                   │                                        ▲
                   └──▶ salience gate ──▶ episodic memory (§F) ──▶ modulator m (§A)
```

One inference step: run bottom-up with current **m** and leaky states; compute bottlenecks; salience gate decides memory writes; memory + pooled state update **m**; top-down feedback re-primes lower blocks; iterate a few steps; emit output. Shims sit inline wherever an override is registered.

---

## 6. Elixir implementation plan

**Stack.** Nx for tensors, Axon for model definition and training, EXLA (XLA) for the compiled backend, Bumblebee where we want to warm-start from a pretrained transformer, GGUF export for inference/distribution (project name). Note a design tension worth resolving early: OTP processes are a tempting literal model for "units/modules as message-passing actors," but training needs dense differentiable tensor ops. Resolution: **actors for orchestration and module composition, Nx/Axon for anything in the gradient path.** Don't put per-unit logic in processes.

**Custom layers to write in Axon:**
- `FiLMModulation` — applies `γ(m)⊙h + β(m)` given the modulator bus.
- `LeakyState` — diagonal SSM/leaky-integrator with learned per-channel `α`; implement the recurrence as an associative scan for parallel training.
- `TypedBottleneck` + `AuxHead` — slice designation plus attached decode heads and their losses.
- `Shim` — adapter/LoRA/steering module reading a bottleneck, trained against a frozen core.
- `EpisodicMemory` — content-addressed store with recency decay and a salience write-gate.
- `FeedbackController` — orchestrates the iterate-to-fixed-point loop with a step cap.

**Milestones (each independently testable):**

1. **Baseline.** Plain Axon transformer + training/eval harness on a small dataset. Establishes the fixed parameter budget for later comparisons. Get GGUF export working end-to-end here so it's not a surprise later.
2. **Temporal state (§B).** Add `LeakyState`; validate on a task needing temporal integration; confirm the multi-timescale variant helps on slow-burn tasks. *Hypothesis 1.*
3. **Modulator (§A).** Add the **m** bus + `FiLMModulation`; show state-dependent priming (deliberately induce and then relax a false-positive bias).
4. **Feedback (§C).** Add `FeedbackController` + top-down projections; measure gains on ambiguous inputs and look for bistability. *Hypothesis 2.*
5. **Structured bottlenecks (§D).** Add `TypedBottleneck`/`AuxHead`; measure attribute decodability and selectivity at cleave points.
6. **Shims (§E).** Freeze the core; train shims; measure edit selectivity (change target attribute, hold others). Stand up the shim registry. *Hypothesis 3.*
7. **Memory (§F).** Add `EpisodicMemory` + salience gate; demonstrate one-shot in-session behavior change that decays as designed. *Hypothesis 4.*
8. **Integration + composition.** Compose multiple shims and a memory-driven modulator; demonstrate a "plug-in module" added to a frozen core without retraining.

Each milestone gates the next; if a hypothesis fails its test, that's a real finding and we branch the plan rather than paper over it.

---

## 7. Training methodology (the notes' staged "cleave and reinforce")

This is the part the notes flag as "much more involved," and it's the real intellectual load.

1. Train (or warm-start) a competent monolithic core on the end task — this doubles as the validation oracle for later stages.
2. Choose cleave points and attribute sets; attach `AuxHead`s and co-train until each attribute is decodable at its bottleneck to target accuracy (§D).
3. Record bottleneck activations across the dataset; freeze the core.
4. Train shims against the frozen core using the recorded interfaces; verify each shim's effect and selectivity against the oracle before admitting it to the registry (§E).
5. Train the salience gate and memory read/write against sequences with noteworthy events; tune recency decay (§F).
6. Train the feedback controller last (or fine-tune jointly with a stop-gradient/implicit-diff scheme) so settling dynamics are stable (§C).

Reusability is the payoff: once a core is cleaved and frozen, new capabilities arrive as shims/memories/modulators that read the existing typed interfaces — no full-path retraining.

---

## 8. Open questions and risks

- **Feedback stability & cost.** Iterating to a fixed point can oscillate or diverge and multiplies compute per step. Needs step caps, implicit differentiation, and a decay-based escape hatch (§C). Highest-risk component.
- **Disentanglement is hard.** §E's selectivity depends on §D actually isolating attributes; perfect disentanglement is famously difficult. Fallback: accept partial selectivity and measure it honestly.
- **Salience gate calibration.** Too eager → memory floods; too shy → nothing written. Likely needs a running-quantile / surprise-normalized threshold.
- **Credit assignment through memory.** Back-propagating through a discrete write-gate needs a relaxation (straight-through / Gumbel) or a policy-gradient treatment.
- **Novelty vs. reinvention.** Several components resemble Mamba (§B), predictive coding / DEQ (§C), and PEFT/steering (§E). We should reuse those directly and keep our contribution to the *composition + cleave-train recipe*, not rebuild each wheel.
- **Elixir ecosystem maturity.** Some custom kernels (associative scans, implicit-diff fixed points) may need EXLA/XLA custom-call work. Validate feasibility in Milestone 2 before committing.
- **GGUF export of nonstandard layers.** Modulation, leaky state, and shims aren't standard GGUF ops; plan for either fusing them into supported ops at export or a custom runtime shim. Confirm early (Milestone 1). *Now planned in full: see [`arch/runtime/`](arch/runtime/README.md) — donor conversion pipeline, llama.cpp deep fork, and the `therobot` GGUF extension spec.*

---

## 9. Glossary

**Modulator / bus (m):** low-dimensional global state (the modulation signals) conditioning the whole network.
**Leaky state:** per-channel decaying memory of recent activation; source of the arrow of time.
**Cleave point / typed bottleneck:** a representation slice required (via aux heads) to encode specified attributes.
**Shim:** an adapter that reads a typed bottleneck and overrides downstream behavior without retraining the core.
**Salience gate:** the surprise/threat/reward trigger that decides when to write an episodic memory.
**Settling:** iterating the forward pass with top-down feedback until interpretation stabilizes.

---

*Origin: `notes.md`, April 9, 2018. This plan preserves the original intent (compact, modular, feedback-driven, faintly mind-like networks) and re-expresses it in terms of mechanisms we can actually build and test today.*
