# AGENT-README

> Orientation for AI coding agents working in this repo. Human docs: README.md.

## Type

python+markdown (design specs in arch/, conversion pipeline in convert/)

## Purpose

The Robot GGUF — transformer/GGUF design and conversion tooling that grafts persistent state, mood/modulation, hot-swappable behavior modules, and episodic memory onto an existing, frozen LLM checkpoint (no retraining; donor tensors untouched).

## Stack

Python pipeline `robotgguf` (`convert/`; numpy + pyyaml, optional HF stack via `.[hf]`), runpod Docker image + helper scripts for rented GPU hosts, design specs/research in `arch/`.

## Dev mode

```bash
cd convert
pip install -e .            # or: pip install -e '.[hf]' for ingest/record/graft
robotgguf --config configs/qwen3.5-0.8b.yaml ingest     # stages: ingest, record, cleave, graft, calibrate, shims, export, verify
```

## Ports / URLs

- none (library/CLI)

## Tests

```bash
cd convert && python -m pytest tests/   # R2/R4/R5/R7/R8 + extraction core run without a GPU
```

## Gotchas

- GPU-bound stages run on rented hosts via the `runpod/` image (model/corpus fetch, volume init).
- Stage outputs append measured values to `<config>.lock.yaml`; the exporter consumes only measured values.
- Converted models stay bit-exact to the donor until the grafted `robot.*` additions are deliberately engaged.
