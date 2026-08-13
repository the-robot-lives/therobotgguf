# Converting Qwen3.5-0.8B — donor-specific instructions

Companion to [`qwen3.5-0.8b.yaml`](qwen3.5-0.8b.yaml). The generic runbook is
the fork's [`docs/robot/validation.md`](../../../3rd-party/llama.cpp/docs/robot/validation.md);
this file covers only what is specific to this donor.

## Why Qwen3.5-0.8B

The Qwen3.5 small series (0.8B / 2B / 4B / 9B, released 2026-03-01) replaced
Qwen2.5 as the capability-per-parameter frontier for self-hosted models. The
0.8B is strictly smaller and newer than the previous Qwen2.5-1.5B target,
instruct-tuned by default, and the fork supports its `qwen35` architecture
natively — the therobot wrapper covers it (one-line family addition, already
landed). Scale-up path once the 0.8B cycle is green: rerun the same config
against Qwen3.5-2B, then 4B, adjusting only the layer lists after R0's survey.

Two donor quirks to keep in mind:

1. **Hybrid layout.** 24 blocks in a 6 × (3×GatedDeltaNet→FFN →
   1×GatedAttention→FFN) pattern — attention blocks at layers 3, 7, 11, 15,
   19, 23; everything else is a linear-attention (recurrent) GDN block. The
   residual stream is uniform (hidden 1024), so taps/shims/FiLM/state all
   work unchanged at `resid_post`, but per-block *dynamics* differ: expect
   R4's Δ statistics to split into two populations (GDN vs attention blocks).
2. **Native recurrence.** GDN blocks already carry state, which makes this
   donor a twofer: it exercises the full graft path *and* the roadmap's
   "SSM-class" claim that §B banks are backbone-agnostic. If the temporal
   suite (R3 gate) shows smaller wins than on a pure-attention donor, that is
   an expected finding, not a failure — the donor already has an arrow of time.

## Step by step

**1. Prepare.** Build the fork and gate binaries and run the fixture suites
green (validation.md Phase 0). On the GPU host:

```bash
cd projects/therobotgguf/convert
pip install -e '.[hf]'
mkdir -p work corpus
# populate corpus/mixed-text.txt (a few hundred MB) + corpus/behavioral-suites.txt
huggingface-cli download Qwen/Qwen3.5-0.8B --local-dir work/hf-qwen3.5-0.8b
```

**2. Stock baseline** (validation.md Phase 1):

```bash
python3 ../../../3rd-party/llama.cpp/convert_hf_to_gguf.py work/hf-qwen3.5-0.8b \
    --outfile work/qwen3.5-0.8b-f16.gguf --outtype f16
# perplexity + eval-suite + latency percentiles on this file = the reference numbers
```

Vet: `llama-robot-inspect work/qwen3.5-0.8b-f16.gguf` reports a stock
`qwen35` file; greedy generations match the HF checkpoint.

**3. Survey + record:**

```bash
robotgguf --config configs/qwen3.5-0.8b.yaml ingest
robotgguf --config configs/qwen3.5-0.8b.yaml record --max-tokens 2000000
```

Vet the lockfile survey: `n_layer: 24`, `n_embd: 1024`, architecture
`qwen3_5`-family. **If the survey disagrees with the YAML's layer lists, fix
the YAML before continuing** — every downstream layer index assumes the 24 ×
hybrid layout. Note: R1's hook helper assumes the `model.model.layers`
layout; confirm the Qwen3.5 module tree matches (adjust `record.py`'s layer
accessor if HF renamed it for the hybrid blocks). Run the weak-labeler pass
to replace placeholder labels before cleave.

**3b. Labels (extraction-v1).** The config now names `semvec:
configs/semvec-v1.yaml`, so labels are the 512-dim vector + categorical
views. With recordings already on disk, everything below re-runs WITHOUT
touching the model:

```bash
robotgguf --config configs/qwen3.5-0.8b.yaml relabel        # t0 vector + views
robotgguf --config configs/qwen3.5-0.8b.yaml labelvec --fit-basis work/semvec-v1-basis.npz
#   → pin latent.basis / basis_sha256 / embedder_revision in semvec-v1.yaml (freezes the standard)
robotgguf --config configs/qwen3.5-0.8b.yaml labelvec --tiers t1        # classifiers + latent block
robotgguf --config configs/qwen3.5-0.8b.yaml labelvec --tiers t2        # teacher (ROBOT_TEACHER_* env)
robotgguf --config configs/qwen3.5-0.8b.yaml labels-qa                  # C2 gates → lockfile
```

Re-baseline note: pre-v1 recordings were drawn head-of-file (extraction-v1
§1.5) — re-record (step 3) with the fixed stratified loader before trusting
any v0-vs-v1 admission comparison.

**4. Cleave:**

```bash
robotgguf --config configs/qwen3.5-0.8b.yaml cleave
```

Vet per validation.md 2.3. Donor-specific expectation: sites at L ≡ 2 (mod 4)
sit directly before attention blocks and typically probe best — if `resid14`
/ `resid18` dominate, that's consistent with the layout, keep 4–6 winners.
With semvec labels present, cleave also runs the vector path
(lockfile `[cleave_vec]`, tensors in `work/semvec-probes/`) — the per-axis
depth map plus proj/calib/overlay per site; `shim-compile` then builds
semvec-defined modules from `semvec_shims:` definitions, and export packages
the readout layer automatically (optional feature; `verify` gate 5 checks it
structurally).

**5. Graft (zero-init first), calibrate, shims:**

```bash
robotgguf --config configs/qwen3.5-0.8b.yaml graft --steps 0
robotgguf --config configs/qwen3.5-0.8b.yaml calibrate
robotgguf --config configs/qwen3.5-0.8b.yaml shims
```

Donor-specific vetting at calibrate: expect bimodal θ across the delta list
(GDN blocks change smoothly; attention blocks jump at retrieval moments). If
attention-adjacent blocks (8, 12, 16, 20 read attention output at 7, 11, 15,
19) show `achieved_keep_rate` far above target, drop them from `delta.layers`
and re-run — a smaller honest delta set beats a noisy large one.

**6. Zero-graft export + the hard gates:**

```bash
robotgguf --config configs/qwen3.5-0.8b.yaml export --out work/qwen3.5-0.8b-therobot-init.gguf
robotgguf --config configs/qwen3.5-0.8b.yaml verify --parity-bin /tmp/robot_parity_test
```

Gate 2 must print `max |logit diff| = 0 … PARITY OK` against
`work/qwen3.5-0.8b-f16.gguf`. Zero means zero — on a hybrid donor this also
proves the wrapper's graph edits splice cleanly around the GDN blocks.

**7. Trained graft, re-export, full vet:**

```bash
robotgguf --config configs/qwen3.5-0.8b.yaml graft --steps 2000   # single-GPU, adapter-scale
robotgguf --config configs/qwen3.5-0.8b.yaml export --out work/qwen3.5-0.8b-therobot.gguf
robotgguf --config configs/qwen3.5-0.8b.yaml verify --parity-bin /tmp/robot_parity_test
../../../3rd-party/llama.cpp/build/bin/llama-quantize \
    work/qwen3.5-0.8b-therobot.gguf work/qwen3.5-0.8b-therobot-q8_0.gguf q8_0
```

Then validation.md Phase 3 Gates 4–7 (toggle matrix, behavioral probes,
serving drill, session lifecycle) on both f16 and q8_0. The priming and
memory probes use the modulator channels declared in the YAML
(`arousal` is channel 0 everywhere in the test templates).

**8. Sign-off + scale-up.** Commit the lockfile, registry, and baseline
numbers together (one provenance unit). When green, copy this config to
`qwen3.5-2b.yaml`, update donor id and the layer lists from that model's R0
survey, and repeat — nothing else should need to change.

## Known limitations against this donor

The delta executor's fire rule compares residual-stream change, which is
well-defined on hybrid blocks, but v1's "blocks still execute" semantics mean
S4's FLOP claims read from the trace (effective FLOPs) — same as any donor.
The settle feature stays off: this is an AR-trained model; `jacobi-ar` would
run but adds nothing for a 0.8B (revisit with the diffusion-class donor).
Multimodal inputs are out of scope — convert and vet text-only.
