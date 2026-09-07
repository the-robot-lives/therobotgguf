# PROJ-ARCH Summary — therobotgguf

## Overview

Retrofits existing LLM checkpoints onto the "therobot" runtime (taps, leaky
state, modulator bus, episodic memory, delta inference, settling decoder).
Docs-plus-tooling repo: design corpus (`arch/`) + Python conversion pipeline
(`convert/`). The runtime itself is an external llama.cpp deep fork. The two
workstreams meet at the versioned GGUF extension spec.

## Pipeline

Stages R0–R8 in `convert/robotgguf` (CLI `robotgguf <stage> --config ...`):
ingest → record → cleave → graft → calibrate → shims → settle (v0 passthrough)
→ export → verify. Prime directive: donor core frozen from recording; grafts
function-preserving at insertion; R8 proves bit-for-bit parity. Config is one
declarative YAML per donor; stages append measured values to a `.lock.yaml`
the exporter alone consumes.

## Runtime model

Capability levels L0–L5 negotiated via `therobot.features`; staged roadmap
S0–S6, each gated by a runnable demo + regression gate. Parity is a permanent
invariant; behavioral probes over benchmarks; latency reported as percentiles.

## Data model

No database — file-based state: donor config YAML → measured lockfile →
recording store (fp16 `.npy` slices + manifest.json) → extended GGUF
(`therobot` / `therobot-shim`) + shim `registry.json`. Versioned contracts,
never patched in place.

## Infrastructure

GPU stages run in the `runpod/` CUDA container (fork build + robotgguf venv);
corpus/checkpoints/work dirs on a persistent volume. Research toolchain, not a
deployed service.

## Stack

Python ≥3.9 (numpy/pyyaml; torch/transformers extras; gguf-py vendored from the
fork), llama.cpp fork in C/C++/CUDA, Modal-hosted teacher endpoint. Elixir
Nx/Axon is the separate ground-up research track.

## Key decisions

Deep fork (fenced ROBOT-EXT insertions, upstream-green superset); measured
values only in exports; versioned contracts; narrow fp16 recording slices;
`robotgguf strip` interop downgrade doubling as the L0 parity fixture.
