# therobotgguf

**Repo:** https://github.com/the-robot-lives/therobotgguf

Ground-up transformer/GGUF design and conversion tooling: grafts persistent state, mood/modulation, hot-swappable behavior modules, and episodic memory onto an **existing, frozen** LLM checkpoint — no retraining, no weight changes to the donor core.

## What

Two halves in one repo:

- **`arch/`** — the therobot runtime design: conversion pipeline spec, runtime architecture, proposals and research notes. The "arrow of time", modulation bus, behavior modules, and episodic memory mechanisms are specified here first.
- **`convert/`** — `robotgguf`, the Python conversion pipeline (workstream C) that retrofits a donor checkpoint (primary target: Qwen3.5-0.8B) onto the therobot runtime and exports an extended GGUF. Donor tensors are untouched; `robot.*` additions start as no-ops, so a converted model's outputs are bit-exact to the donor until deliberately engaged.
- **`runpod/`** — Docker image + helper scripts for rented GPU hosts (model/corpus fetch, volume init).

## Why

Bigger models cost more and improve only by expensive retraining. The premise: a compact model's ceiling is set less by parameter count than by missing *mechanisms* — state across tokens, steering beyond prompting, session memory. Grafting those onto a model you already have is a different lever than scaling up. Informs and couples to the Noizu `Libs/ai` (genai) family.

## Getting Started

Work happens in `convert/` (the only build system at present):

```bash
cd convert
pip install -e .            # numpy + pyyaml; pip install -e '.[hf]' for ingest/record/graft (HF stack)
robotgguf --config configs/qwen3.5-0.8b.yaml ingest    # R0  (needs HF stack + GPU)
robotgguf --config configs/qwen3.5-0.8b.yaml record    # R1
robotgguf --config configs/qwen3.5-0.8b.yaml cleave    # R2
robotgguf --config configs/qwen3.5-0.8b.yaml graft     # R3  (--steps N trains; 0 = zero-init)
robotgguf --config configs/qwen3.5-0.8b.yaml calibrate # R4
robotgguf --config configs/qwen3.5-0.8b.yaml shims     # R5
robotgguf --config configs/qwen3.5-0.8b.yaml export --out work/model.gguf  # R7
robotgguf --config configs/qwen3.5-0.8b.yaml verify    # R8 parity gate
robotgguf strip work/model.gguf stock.gguf             # downgrade for stock llama.cpp
```

End-to-end tests (R2/R4/R5/R7/R8 + extraction core) run **without a GPU**: `python -m pytest tests/` from `convert/`. GPU-bound stages run on a rented host via the `runpod/` image. Stage outputs append *measured* values to `<config>.lock.yaml`; the exporter consumes only measured values.

## How It Works

- **Prime directive:** donor core frozen at recording; every graft is function-preserving at insertion; the `verify` stage proves bit-parity against the live runtime — checked, not asserted.
- **Stages R0–R8:** ingest (HF checkpoint) → record activations → relabel/labelvec (semvec readout layer: stratified corpus, tiered labelers) → cleave (per-site ridge maps + readout/overlay pairs) → graft (zero-init trainable modules) → calibrate → shims → settle → export → verify.
- **extraction-v1** adds the corpus-scale vectorized extraction track (`semvec/v1` spec, vector cleave, semvec-defined shims) — see `extraction-v1.md`.
- The exported GGUF is a strict superset: donor tensors unchanged, plus dormant `robot.*` additions; `strip` removes them for stock llama.cpp interop.

## Docs

- `overview.md` — executive summary + engineering deep-dive
- `extraction-v1.md` — semvec/extraction track design
- `docs/PROJ-ARCH.md`, `docs/PROJ-LAYOUT.md`, `docs/PROJ-SCHEMA.md`
- `convert/README.md` — full pipeline usage and stage status table
- `arch/runtime/conversion-pipeline.md` — the spec the pipeline implements
