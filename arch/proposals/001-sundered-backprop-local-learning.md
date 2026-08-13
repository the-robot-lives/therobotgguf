# 001 — Sundered Backprop: Module-Parallel Local Learning

**Status:** proposal
**Primary targets:** faster training (wall-clock), biologically-motivated parallel *learning*
**One-line thesis:** Kill the global backward pass. Train the network as a mesh of modules that learn simultaneously from locally available signals, with typed bottlenecks (planning §D) serving as the contracts between them.

---

## 1. The bottleneck we're attacking

End-to-end backpropagation serializes training three ways (the DNI paper's "locking" taxonomy):

- **Forward lock** — layer L+1 can't start until L finishes.
- **Update lock** — no layer can update until the *entire* forward pass completes.
- **Backward lock** — layer L can't update until every layer above it has computed its gradient.

Consequences: depth-sequential wall-clock, O(depth) activation storage held hostage for the backward pass, and global synchronization barriers that pipeline parallelism only papers over with bubbles.

Biological learning does none of this: there is no global loss broadcast from a distal output back through every connection. Each region adjusts from *local* information — local pre/post activity plus diffuse modulatory signals. That third factor is exactly the modulator bus **m** we already planned (§A).

## 2. The overhaul

1. **Partition at cleave points.** The typed bottlenecks (§D) become module boundaries. Modules M₁…Mₖ each own their optimizer, their device, and their lifecycle.
2. **Local objective per module,** a weighted sum of:
   - **Predictive-coding loss** — the top-down predictions from §C's feedback pathway, repurposed as the *teacher*: each module minimizes the error between its output and what the module above predicted it should be. (Predictive coding provably approximates backprop under mild conditions — Whittington & Bogacz; Millidge et al.)
   - **Aux-head decodability** — the §D deep-supervision losses, already in the plan.
   - **Synthetic-gradient critic (optional)** — a small learned model of the downstream gradient (DNI-style) to keep modules aligned with the end task without waiting for it.
   - **Local self-supervision** — contrastive or forward-forward term where no downstream signal exists yet.
3. **Asynchronous training loop.** Modules exchange bottleneck activations (upward) and predictions (downward) as messages. Each module steps on whatever it has, tolerating bounded staleness. No global barrier anywhere.
4. **Modulated learning.** **m** gates each module's learning rate: high-surprise state → high learning rate, calm → consolidating. This is the classic three-factor learning rule, and it reuses §A unchanged.

Note the inversion worth savoring: planning.md treats top-down feedback (§C) as an *inference* mechanism with training-stability risk. Here feedback **is the training signal**. The thing that was the risk becomes the teacher.

## 3. Training mechanics

- **Forward unlock:** module i processes batch t while module i+1 is still on batch t−1. Staleness is bounded (e.g., ≤2 steps) and EMA-smoothed targets absorb the jitter.
- **Memory:** no full-depth activation retention. Each device holds only its own module's activations → the freed memory buys larger batches or wider modules.
- **Elixir/OTP fit:** this is the planning.md resolution ("actors for orchestration, Nx for the gradient path") scaled up to training. Each module is a supervised GenServer owning an Nx/EXLA graph; bottleneck tensors are the messages; BEAM distribution across nodes is the interconnect. Only the §D slices cross process boundaries — which are *small by design*, because cleaving already forces compact typed interfaces. The architecture decision and the systems decision turn out to be the same decision.

## 4. Inference

Unchanged — this proposal is training-side. But one durable payoff: because no module ever depended on gradients flowing through its neighbors, any module can later be retrained, replaced, or upgraded in isolation. That is the load-bearing precondition for Proposal 005 (frozen-core accretion).

## 5. Mapping to planning.md components

| Component | Role in this proposal |
|---|---|
| §A modulator **m** | Third factor gating per-module learning rate |
| §B leaky state | Stays module-local; no cross-module BPTT needed |
| §C top-down feedback | Becomes the predictive-coding teaching signal |
| §D typed bottlenecks | Module boundaries = training contracts = message schema |
| §E shims | Trained with exactly this machinery (local loss vs frozen neighbor) |
| §F salience gate | Surprise signal that drives the learning-rate gate |

## 6. What it buys (honest arithmetic)

- Backprop wall-clock ≈ (forward + backward over full depth), globally synchronized. Local-mesh wall-clock ≈ the slowest module's local step. With k balanced modules: up to k× depth-parallelism *on top of* data parallelism, minus messaging overhead.
- Activation memory drops from O(total depth) to O(module depth) per device.
- **Known cost:** greedy/local learning historically lands below end-to-end BP on hard tasks. Belilovsky et al. narrowed the gap to ~1–2% on ImageNet-scale; predictive-coding variants close it further at higher compute. Treat the gap as a measured quantity, not a hope.

## 7. Falsifiable tests

1. **Parity band:** fixed-parameter LM trained as a local mesh lands within 5% eval loss of the Milestone-1 backprop baseline.
2. **Scaling:** wall-clock vs device count beats pipeline parallelism beyond the point where bubbles dominate (measure the crossover).
3. **Contract stability:** retrain module k alone against its recorded interfaces; neighbors untouched; end-task metrics hold. (This test doubles as the admission gate for Proposal 005.)

## 8. Risks & mitigations

- **Greedy features** that serve the local loss but starve downstream modules → keep the synthetic-gradient critic, plus an occasional cheap global fine-tune "annealing pass."
- **Staleness divergence** → bound staleness, EMA target networks, per-module gradient clipping.
- **BEAM messaging overhead** → only bottleneck slices cross processes; keep tensors on-device via EXLA device pointers; batch messages.
- **Predictive-coding ≈ BP only under assumptions** → verify empirically at small scale before scaling (Milestone-5/6 pilot, see below).

## 9. Prior art to build on

Forward-Forward (Hinton 2022); predictive coding as BP approximation (Whittington & Bogacz 2017; Millidge et al. 2020); Decoupled Neural Interfaces / synthetic gradients (Jaderberg et al. 2017); greedy layerwise scaling (Belilovsky et al. 2019); local losses / decoupled parallel training (LoCo and successors); three-factor modulated learning rules.

## 10. Fit with the milestone plan

Pilot after Milestone 5 (cleave points must exist to define module boundaries). Run the mesh trainer as an *alternative* trainer for the same architecture — the Milestone-1 harness gives the BP baseline for the parity test for free. If the parity test fails by >5%, keep BP for the core and retain local learning only for shim training (where it is already the natural fit).
