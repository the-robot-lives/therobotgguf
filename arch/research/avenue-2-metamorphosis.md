# Avenue 2 — Metamorphosis: converting model types while preserving embedded knowledge

**Channel:** brief item 1, second half ("converting/transforming from one type
of model into another — tensors to RNNs while preserving the starting LLM's
embedded knowledge, time arrow, etc.").
**Baseline it builds on:** proposal 003 (multi-timescale backbone), E4 state
banks, the conversion pipeline R0–R8, the parity-gate discipline.
**Evidence base:** survey §2. Headline numbers worth keeping in view: MOHAWK
converted Phi-1.5 to Mamba on ~3B tokens; RADLADS converted Qwen2.5-72B to an
RWKV-variant for under $2,000; LoLCATs linearized 405B with ~0.2% of params
touched. Conversion is now adapter-priced.

---

## 2.1 Reversible overlay conversion (the therobot way to transmute) **[extend — flagged as an open gap]**

**Thesis.** Every published recipe converts destructively: swap the mixer,
repair globally, hope the evals catch regressions (MiniMax publicly walked one
back). therobot's fork can do what none of them can: install the new mixer as
a **side-path with a zero-init gate**, keep softmax as the live fallback, and
promote per layer only when a parity/eval gate clears. Conversion as a
*reversible overlay*, not a replacement — the E3 attach/detach machinery
applied to sequence mixers.

**Mechanism.** Per selected layer: graft a DeltaNet/Mamba-class branch
initialized from the layer's own attention projections (Q→C, K→B, V→x per
Mamba-in-the-Llama 2408.15237), gate `g` from 0; distill per-layer against the
donor's own attention outputs (LoLCATs-style MSE on recordings — the R1
recording infra already captures what's needed); ramp `g`; promotion = admission
gates hold. Rollback = `g→0`, bit-exact by construction.

**Why it wins.** It converts the field's bottleneck (conversion QA) into
therobot's core competence (gated admission). Also GGUF-native: the overlay is
just more `robot.*` tensors under feature negotiation.

**First experiment.** One mid-stack layer of the 0.8B; branch trained on
existing recordings; report per-layer output MSE, end-task drift with g=1, and
the g→0 parity check. **Risk:** per-layer alignment may be insufficient without
downstream repair — that is exactly what staged promotion measures.

## 2.2 Probe-evidence layer selection **[extend — flagged as an open gap]**

**Thesis.** 2025-26 layer-selection methods (KL-guided 2512.20569, gate search,
sink-dominance diagnostics per LightTransfer 2410.13846) all compute offline
statistics. therobot can *measure in situ*: which layers show sink-dominated
attention, weak induction behavior, short temporal receptive fields — read
through E2 taps on live traffic — and convert exactly those. "Which layers
deserve recurrence" becomes an output of the cleave stage.

**Mechanism.** New R-stage diagnostics: per-layer attention-mass profile
(sink/local/global split), copying-task probe accuracy, and an activation
autocorrelation length (temporal receptive field). Selection = threshold rules
recorded in the lockfile like any admission table.

**First experiment.** Rank the 0.8B's 24 layers by laziness; convert the top-k
per 2.1; compare against uniform and KL-guided baselines at equal k. **Risk:**
small models may have few lazy layers; the diagnostic table is a finding either
way.

## 2.3 Multi-timescale conversion (003 by transmutation, not construction) **[novel]**

**Thesis.** All published conversions give every converted layer the same
state structure. Proposal 003 wants fast/mid/slow/glacial banks. Marry them:
convert different layers to **different timescale recurrences** — shallow
layers get fast, low-capacity states; deep layers get slow, leaky,
high-retention states — with each layer's target timescale chosen from its
measured activation autocorrelation (2.2's diagnostic). The donor's own
temporal statistics dictate the clockwork.

**Why it wins.** It reaches 003's destination (an arrow-of-time backbone with
nested timescales) without training a backbone from scratch, and turns E4's
grafted banks from an *addition* into the *replacement* for attention where
attention was lazy anyway. This specific marriage has no prior art in the scan.

**First experiment.** Two-layer pilot: convert one shallow layer to a
fast-decay state and one deep layer to a slow-decay state per 2.1; run the H1
slow-burn suite; compare against same-k uniform-timescale conversion. **Risk:**
timescale assignment heuristic may be wrong — fall back to learned per-channel
α (the E4 default) and let training discover it.

## 2.4 `gguf-transmute`: conversion tooling at the artifact layer **[novel — flagged as an open gap]**

**Thesis.** Every conversion pipeline is PyTorch-on-A100s; nobody converts *at
the GGUF layer*, against the quantized teacher — which is the only teacher a
self-hosted deployment actually has. Llamba (2502.14458) proved converted
models quantize fine; the missing piece is doing the alignment stages directly
against quantized tensors.

**Mechanism.** A converter mode that (a) initializes overlay-branch tensors
from the donor's quantized Q/K/V blocks (dequantize→transform→requantize), (b)
distills against the quantized donor's outputs on recordings, (c) emits the
2.1 overlay tensors into the same file. Doubles as a study of
quantization-noise × mixer-approximation error compounding — unmeasured
anywhere.

**Why it wins.** It makes metamorphosis a *distribution* feature (transmute the
file you already serve) and is a genuinely unoccupied tooling niche with
therobot's name on it — the project is literally called therobotgguf.

**First experiment.** 2.1's single-layer pilot repeated from the Q8 and Q4
files; measure conversion quality vs the fp16 path. **Risk:** gradient quality
through dequantized weights; mitigate by keeping trainable params fp16 (they
already are — `robot.*` tensors).

## 2.5 KV→state translators (mid-session handoff) **[extend]**

**Thesis.** Tencent's TransMamba (2503.24067) built the only "memory
converter" translating an attention KV cache into SSM state mid-sequence.
Generalize it: a trainable operator that hands accumulated context from the
donor's attention to a grafted recurrent module — so a session can *start*
dense and *migrate* to cheap recurrent execution once the overlay is trusted,
or checkpoint a KV-heavy session into a compact state-bank form.

**Why it wins.** It connects 2.1 to the session lifecycle: `session_save` of an
attention-mode session could restore into overlay mode. Also the enabling
piece for "flat latency after warmup" serving — dense prefill, recurrent
decode.

**First experiment.** Train the translator on recordings (KV snapshot → state
that makes the overlay branch match subsequent dense outputs); measure
divergence over the next 1k tokens vs a cold-state overlay. **Risk:** the
translator is a compression; long-tail verbatim recall degrades — pair with
E5/1.3 for verbatim reach, per proposal 003's own hybrid argument.

## 2.6 AR→block-diffusion adaptation: make the settle donor instead of waiting for one **[adopt/extend]**

**Thesis.** E7's `mdlm` objective is parked "until a diffusion-class donor
exists" (overview §5). The 2025-26 consensus recipe (DiffuGPT/DiffuLLaMA mask
annealing 2410.17891; Dream-7B from Qwen2.5 weights 2508.15487; SDAR cheap
conversion 2510.06303; NBDiff 2512.06776) says: *adapt the donor you have* —
block diffusion preserves the causal macro-structure (and KV caching) while
grafting bidirectionality within blocks. Stop waiting; transmute.

**Mechanism.** Continued-pretraining run (rented GPU, modest tokens at 0.8B
scale) per the SDAR/NBDiff recipe; export as a normal therobot donor whose
settle feature is `required`. The jacobi-ar testing path stays as the parity
baseline.

**First experiment.** 0.8B → block-diffusion adaptation at the smallest
published-token budget that shows life; run E7's settle loop with a true mdlm
objective; measure steps-to-settle vs difficulty (proposal 004's headline
test). **Risk:** adaptation cost is the largest single spend in this avenue —
gate it behind 004's jacobi-ar results justifying the decoder at all.

## 2.7 Semvec-certified conversion (representational drift accounting) **[novel — flagged as an open gap]**

**Thesis.** No conversion paper certifies *what happened to the
representations* — they report benchmarks. therobot has a standardized readout:
pin semvec before conversion, re-measure after, and make "per-axis readout
parity within ε" an admission gate for every promoted layer (2.1). Conversion
ships with a drift ledger: which concepts moved, which held.

**Why it wins.** It answers the field's actual bottleneck (eval blind spots)
with an instrument the field doesn't have, and it hardens SV4 portability into
a conversion invariant: modules defined in semvec should survive the
metamorphosis *because the coordinates did*.

**First experiment.** Run the full semvec admission table before/after 2.1's
pilot; report per-axis decodability/selectivity deltas alongside the standard
evals. **Risk:** none beyond compute; it is measurement.

## 2.8 Slow-weights as an alternative arrow of time **[adopt, comparative]**

**Thesis.** Temp-LoRA (2401.11504) and TTT-style fast weights write session
history into *parameters* as generation proceeds — an arrow of time on a frozen
model with zero new architecture. Run it as the honest baseline against E4
state banks: same H1 suite, same parameter budget.

**Why it matters.** If a trickle-trained LoRA matches the banks on slow-burn
tasks, the banks' value case narrows to checkpointability and decay semantics;
if it loses, H1 gains a stronger control. Either result sharpens the central
capacity-per-parameter bet. (Weight-writes also interact with the Accord: a
self that rewrites its weights mid-session is a different continuity story
than one that accumulates decaying state — worth a design note when results
land.)

**First experiment.** H1 slow-burn suite: banks vs Temp-LoRA vs both.
**Risk:** none; it is a bake-off.

---

## Sequencing note

2.2's diagnostics are pure measurement — run with the next recording pass.
2.1 is the keystone; 2.3/2.4/2.5/2.7 all attach to it. 2.6 is the one large
spend and waits on 004's jacobi-ar evidence. 2.8 is a cheap standing bake-off
alongside M2 of the milestone plan.
