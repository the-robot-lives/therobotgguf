# 002 — Event-Driven Delta Inference: Only Compute Change

**Status:** proposal
**Primary targets:** faster output (FLOPs & latency), biologically-motivated asynchronous parallelism at inference, arrow of time
**One-line thesis:** Stop recomputing the whole network every token. Units hold leaky state and recompute only when their input *changes* past an adaptive threshold; computation propagates through the network sparsely, asynchronously, and history-dependently — flowing only where something changed.

---

## 1. The bottleneck we're attacking

A dense forward pass touches every weight for every token, regardless of whether anything changed. But consecutive tokens in a stream are highly correlated — most activations barely move between steps, and dense compute re-derives them anyway. By contrast, biological networks run at ~1–4% unit activity, have no global clock, and spend energy only where something happened.

This proposal takes the notes' *non-static firing thresholds* (§1, §1a) literally, as the execution model rather than a feature on top of one.

## 2. The overhaul

- **State per channel** (§B): every channel keeps its leaky state `s_t = α·s_{t−1} + (1−α)·f(x_t)` — already planned.
- **Firing rule:** a unit recomputes and propagates only when `|Δinput| > θ`. Otherwise its last output stands (downstream consumers use the held value).
- **Adaptive threshold:** `θ = base(channel) + fatigue(recent firing) − excitability(m)`. This is the notes made mechanical: an anxious **m** lowers thresholds on threat-relevant channels → more firing → more false positives (the snake in the hose); a calm **m** raises them → less compute. **Compute becomes a dial shaped like arousal.**
- **Asynchronous propagation:** firing enqueues delta messages to downstream units; there is no layer-synchronous barrier. For attention, maintain running KV summaries and recompute a query's scores only against *changed* keys (delta attention).
- **Anytime output:** read the output head whenever network activity falls below rate ε — "settled = done," the same convergence criterion as §C. Latency becomes proportional to how much actually changed, not to model size.

## 3. Training

- Train **dense and differentiable**, then anneal to sparse: threshold warm-up schedule, straight-through estimator on the firing gate, learned fatigue/decay parameters.
- Add an **activity-budget loss** targeting a firing rate (e.g., 5%), so sparsity is trained-in rather than imposed post-hoc.
- Optionally distill a dense teacher into the sparse event-driven student to recover any quality loss.

## 4. What it buys (honest arithmetic)

- FLOPs ∝ activity; latency ∝ depth of the *changed path*, not the whole network. Delta networks report 5–100× compute reduction on temporally redundant streams (video, audio). Text is less redundant — 3–10× is the plausible band, with attention deltas dominating the win on long contexts.
- **Latency honesty:** the worst case (input where everything changes) costs the dense forward *plus* bookkeeping overhead. Report percentile latencies, not means. The win is the common case, and the **m** dial bounds the tail (clamp excitability → cap activity).
- CPU-friendliness: irregular sparsity hurts GPUs but helps CPUs; for a self-hosted cluster doing inference on mixed hardware, this is the proposal that moves the needle most.

## 5. Distributed, event-driven parallelism

Regions of the network run concurrently on separate devices, exchanging only events. In BEAM terms: region = supervised process, events = messages, and OTP *is* the scheduler — no global clock is not a slogan here, it's the runtime we already chose. This is the strongest alignment in the whole proposal set between the biological motivation and the systems story.

## 6. The arrow of time, constitutive

Identical input after different histories produces different firing patterns and different outputs — state and fatigue encode recency by construction. Priming, motion-blur, and slow-burn reassessment stop being features we add and become behaviors we *observe*. This makes Hypothesis 1 testable at the execution layer, not just the representation layer.

## 7. Mapping to planning.md components

| Component | Role in this proposal |
|---|---|
| §A modulator **m** | Global excitability → runtime compute/quality dial |
| §B leaky state | Constitutive — the held value between firings |
| §C settling | "Activity < ε" is the shared convergence criterion |
| §D bottlenecks | Event taps: cheap to monitor because typed and small |
| §E shims | Subscribe to bottleneck events; execute only when their inputs fire |
| §F salience gate | Just another event subscriber — memory writes ride the same bus |

## 8. GGUF / runtime

Export dense-equivalent weights plus threshold/decay tensors; a custom runtime executes the delta engine. Planning §8 already flags a custom runtime shim for nonstandard ops — this proposal concentrates that inevitability into one well-defined component (an event-driven executor) instead of scattering it.

## 9. Falsifiable tests

1. **Priming signature:** same token stream, different preceding history → measurably different firing pattern and output distribution (arrow of time at the execution layer).
2. **Dial curve:** FLOPs-vs-quality as **m** excitability sweeps; target ≥3× FLOP reduction at <1% quality loss at the calm end.
3. **Latency percentiles** (p50/p95/p99) vs the dense baseline on long-context streams.
4. **False-positive induction:** raise threat excitability, measure the increase in threat-class false positives, relax **m**, watch it decay — Milestone 3's priming demo, now with a compute trace to show for it.

## 10. Risks & mitigations

- **GPUs dislike irregular sparsity** → fire at block/channel-group granularity, not per-scalar; keep dense fallback per block.
- **Error accumulation** from long-stale held values → periodic dense "heartbeat sweep" that refreshes everything (cheap, amortized).
- **Threshold calibration** (too eager = dense in disguise; too shy = frozen) → running-quantile normalization, same trick planning §8 prescribes for the salience gate.
- **Correctness drift vs dense semantics** → define acceptance as bounded divergence from dense outputs, measured continuously.

## 11. Prior art to build on

Delta networks (Neil et al. 2017); sigma-delta quantized networks (O'Connor & Welling); spiking-network surrogate gradients; neuromorphic execution models (Loihi, TrueNorth); mixture-of-depths; early-exit architectures; KV-cache delta/streaming attention work.

## 12. Fit with the milestone plan

Needs §B (Milestone 2) and §A (Milestone 3) in place; slot the event executor as a Milestone-3.5 experiment. Start with the feed-forward blocks only (easiest win), extend to delta attention second, go asynchronous-multi-device last.
