# Conversion Pipeline — Retrofitting Existing LLMs onto the therobot Runtime

**Status:** plan
**Stack:** Python (HF Transformers, PyTorch, PEFT-style training, `gguf-py` emission)
**First donor:** Llama/Qwen-class 1–4B instruct checkpoint (e.g. Qwen2.5-1.5B/3B, Llama-3.2-1B/3B)
**Prime directive:** the donor core is **frozen** from the moment of recording (proposal 005). Every added parameter is either trained against recordings or trained with the core frozen, and every graft is **function-preserving at insertion** (proposal 006's zero-effect nascent-unit trick). A converted model with all extensions at init is bit-for-bit the donor.

---

## 1. Package layout

```
projects/therobotgguf/
  convert/
    pyproject.toml
    robotgguf/
      cli.py            # `robotgguf <stage> --config configs/<donor>.yaml`
      config.py         # conversion config schema (the per-donor YAML)
      ingest.py         # R0
      recordings.py     # recording store (write/read, manifests, hashing)
      record.py         # R1
      cleave.py         # R2
      graft.py          # R3 (modules + training loop)
      calibrate.py      # R4
      shims.py          # R5
      settle.py         # R6 (separate track)
      export.py         # R7 (gguf-py emitter)
      verify.py         # R8 (parity + toggle harness; drives the runtime binaries)
    configs/
      qwen2.5-1.5b.yaml
```

The per-donor YAML is the single declarative artifact: donor id, calibration corpus, candidate cleave sites, attribute set, bank layout, thresholds, export options. Every stage reads it; stages append their measured outputs (chosen sites, probe scores, calibrated θ) back into a lockfile (`<donor>.lock.yaml`) so downstream stages and the exporter consume *measured* values, never hand-entered ones.

## 2. Stages

### R0 — Ingest & survey
Load the HF checkpoint, emit a survey report: architecture, layer count, dims, norm placement, candidate cleave sites (default: residual stream after each block's FFN, every 4 blocks), tokenizer facts, licensing note. Sanity-run generation. Output: survey section of the lockfile.

### R1 — Record
Run the calibration corpus (a few hundred MB of mixed text + the project's behavioral task suites) through the frozen donor with forward hooks on candidate sites. Store **slices only** (candidate widths, e.g. 64–256 channels of the residual stream), fp16, sharded and memory-mapped, with a manifest binding `{model hash, corpus hash, spec version}` — recordings are versioned contracts (005 §6) and regenerable derived data. This recording store is the training substrate for **everything** downstream: probes (R2), shims (R5), thresholds (R4), and later accreted modules.

### R2 — Cleave (typed bottlenecks, §D)
The notes' vision-flavored attributes (subject/color/size) need a text-domain v0 set. Start coarse and decodable:

- **language**, **register/formality**, **sentiment/valence**, **topic domain** (coarse taxonomy), **speech-act** (question/command/statement), **safety/threat salience**, **entity-presence**.

Labels come from weak labelers (small classifiers / teacher-LLM annotation over the calibration corpus — label once, reuse forever). For each candidate site × attribute: train a small probe (linear, then 1-hidden-layer if linear fails) on recordings. Score:

- **decodability** — probe accuracy/F1;
- **selectivity** — margin over the same probe trained on a random equal-width slice at the same layer;
- **stability** — score variance across corpus shards.

Select final cleave points (target: 4–8 bottlenecks over the depth), write `bottleneck` entries + admission scores to the lockfile, and export the winning probes as GGUF probe tensors. **No core fine-tuning in the retrofit path** — attributes that aren't decodable in the frozen donor are dropped (recorded as findings), not forced via deep supervision; that option belongs to the ground-up track.

### R3 — Graft (leaky state §B + modulator §A)
Wrap the HF model in a `RobotModel` that inserts, per covered block:

- **LeakyState branch:** `s_t = α·s_{t−1} + (1−α)·f(h_t)` with per-channel learned `α` (parameterized as logits over banks: fast/mid/slow/glacial per 003 §2), input proj `f`, and an output projection **initialized to zero** — the graft contributes nothing at insertion.
- **FiLM heads:** `h' = γ(m)⊙h + β(m)` with `γ≡1, β≡0` at init (identity).
- **Modulator bus:** pooled activations (+ later, memory recall) → small GRU/MLP → `m` (dim 8–32, named channels per planning §A), with per-channel decay so `m` relaxes to baseline.
- The **glacial bank is `m`'s source** (003's unification) — the slowest state stripe feeds the modulator update rather than being a separate module.

Training: core frozen; new params only (adapter-scale, single-GPU-days at 1–4B):
1. **Distillation anchor** — KL to the frozen donor's logits on plain text, so grafts learn to help without drifting the base distribution.
2. **Temporal suite** — synthetic long-range/slow-burn tasks (the "polite stranger" pattern: evidence accumulates across many tokens and must flip an interpretation late — 003 test 4) to give the state banks a reason to exist.
3. **Priming suite** — condition `m` channels, verify controllable bias appears and decays (planning M3's induce-and-relax false-positive demo).

Gate: with grafts at init, logit parity with the donor; with grafts trained, distillation-anchored perplexity within a stated band and temporal-suite wins over the ungraftedd donor.

### R4 — Calibrate (delta thresholds, 002)
No training required for v0. From the recordings (plus a fresh streaming pass), compute per-block (later per-channel-group) statistics of `|Δinput|` between consecutive tokens. Set `θ_base` by running-quantile at a target keep-rate (e.g. 30% block-executions on streaming text), emit fatigue/excitability parameters at neutral defaults, and validate **bounded divergence**: delta-mode outputs vs dense on held-out streams within tolerance, with the heartbeat sweep interval chosen from the divergence curve. Optional later: anneal thresholds with a straight-through gate + activity-budget loss (002 §3) for a better FLOPs/quality frontier.

### R5 — Shims (§E) + salience (§F)
Shims train **against recordings only — the core is not loaded** (005 §2.3): slice-scoped LoRA/steering modules reading a bottleneck and emitting a modified slice. v0 shim set: register/formality steering, topic bias, safety damping. Admission per 005 §2.4: selectivity score (move the target attribute, hold the others — measured with R2's probes), regression vs the donor oracle, composition check against already-admitted shims. Admitted shims are exported as standalone module GGUFs plus a registry entry.

Salience gate calibration: surprise signal = logprob spike + `m` magnitude, threshold set by running quantile (planning §8's prescription) on the calibration streams; summary head = the bottleneck slices themselves projected to the memory key dim (reuse, don't invent).

### R6 — Settling track (004) — separate, optional, expensive
Converting an AR donor to a masked-diffusion objective is a large continued-pretraining job, not a retrofit step. Resolution:
- **v0:** run an *existing* diffusion checkpoint (Dream/LLaDA-class) through R0–R5 — the extensions are objective-agnostic — and let the runtime's settle decoder (E7) drive it. This ships L4 without a training moonshot.
- **Later experiment:** AR-donor → MDLM adaptation (Dream-style AR-init recipe) at 1–4B scale, budgeted and gated separately.

### R7 — Export
Emit the extended GGUF per [`gguf-extension-spec.md`](gguf-extension-spec.md): stock base tensors (converted via llama.cpp's own `convert_hf_to_gguf.py`, vendored from the fork) + `robot.*`/`blk.*.robot_*` extension tensors + `therobot.*` metadata from the lockfile. Also: shim module files, and `robotgguf strip` (extended → stock base file) for interop. Quantization: base tensors quantize as usual via the fork's `llama-quantize`; extension tensors are small and stay f16/f32.

### R8 — Verify
Drives the runtime binaries end-to-end; this stage owns the gates in README §5: L0 parity, per-feature toggle matrix (perplexity + eval suite + latency percentiles), behavioral probes (priming, memory decay, delta divergence, settle-difficulty correlation), and shim selectivity re-checks against the *live* runtime (not just recordings).

## 3. What retrofit deliberately does NOT do

- No core fine-tuning, ever (that's the ground-up Elixir track's business — and 005's one unforgivable sin is patching a frozen core in place).
- No deep-supervised cleave training — retrofit *discovers* bottlenecks in the frozen donor; it doesn't force them.
- No unit lifecycle (006) — lifecycle is a training-time property of the ground-up track; retrofitted exports are static graphs by construction.

## 4. Donor roadmap

1. **Qwen/Llama-class 1–4B instruct** — full pipeline, all stages except R6.
2. **Mamba/RWKV-class** — §B is native (banks map onto existing SSM state; the graft stage shrinks to modulator+FiLM only); validates the spec's claim that banks are backbone-agnostic.
3. **Dream/LLaDA-class diffusion checkpoint** — the S5 settle donor.
4. Later: 7–8B of whichever family won on quality-per-watt for the self-hosted cluster.
