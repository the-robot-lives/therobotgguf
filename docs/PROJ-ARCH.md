# Project Architecture — therobotgguf

## Overview

therobotgguf retrofits *existing* LLM checkpoints onto the "therobot" runtime —
a transformer augmented with typed bottleneck taps, multi-timescale leaky state,
a modulator bus, episodic memory, delta inference, and a settling decoder. It is
a docs-plus-tooling project: the runtime itself is a **deep fork of llama.cpp**
maintained elsewhere (all extension code in `llama-robot-*` files, insertions
fenced with `ROBOT-EXT` markers); this repo owns the **design corpus** (`arch/`,
root `*.md`) and the **Python conversion pipeline** (`convert/`) that turns an HF
checkpoint into an extended GGUF the fork loads.

Two interlocking workstreams, defined in `arch/runtime/README.md`:

- **Workstream C — conversion** (`convert/`): `robotgguf`, stages R0–R8.
- **Workstream R — runtime**: the llama.cpp fork plan (`arch/runtime/llamacpp-extensions.md`).

They meet at the **GGUF extension spec** (`arch/runtime/gguf-extension-spec.md`)
— the single versioned contract both sides implement.

## System diagram

```mermaid
graph LR
    subgraph inputs
        HF[HF donor checkpoint]
        CORPUS[corpus/manifest.yaml]
    end
    subgraph robotgguf CLI — convert/
        R0[R0 ingest] --> R1[R1 record]
        R1 --> R2[R2 cleave / labelvec]
        R2 --> R3[R3 graft]
        R3 --> R4[R4 calibrate]
        R4 --> R5[R5 shims / shim-compile]
        R5 --> R6[R6 settle v0]
        R6 --> R7[R7 export]
        R7 --> R8[R8 verify]
    end
    REC[(recording store\nwork/recordings)]
    LOCK[(<config>.lock.yaml\nmeasured values)]
    CFG[configs/<donor>.yaml]
    GGUF[extended GGUF\narch therobot]
    SHIM[shim GGUFs + registry.json]
    FORK[llama.cpp fork\nllama-robot-*]

    CFG --> R0
    HF --> R0
    CORPUS --> R1
    R1 --> REC
    REC --> R2 & R4 & R5
    R0 & R2 & R4 --> LOCK
    LOCK --> R7
    R7 --> GGUF
    R7 --> SHIM
    R8 --> FORK
    GGUF --> FORK
```

## Core components

| Component | Purpose |
|-----------|---------|
| `convert/robotgguf/` | CLI + one module per stage (R0 ingest → R8 verify) plus config/lockfile, corpus loader, labeler tiers, semvec readout |
| `convert/configs/` | Per-donor declarative YAML (single source of config) + measured `.lock.yaml` state + `semvec-v1.yaml` (versioned standard) |
| `arch/` | Design docs: staged plan, GGUF extension spec, fork plan, proposals 001–006, research avenues |
| `runpod/` | CUDA container for GPU-bound stages (ingest/record/training) with volume-persistent data dirs |
| llama.cpp fork *(external)* | Runtime: `therobot` arch family, taps/shims/state/FiLM/memory/delta/settle execution, `llama-robot-inspect` for R8 |

## Conversion pipeline (Workstream C)

**Prime directive** (proposal 005): the donor core is *frozen from the moment of
recording*; every graft is function-preserving at insertion (zero-init outputs,
γ≡1/β≡0 FiLM); R8 proves a converted model with extensions at init is
**bit-for-bit the donor** — checked, not asserted.

| Stage | Does |
|-------|------|
| R0 ingest | Load HF checkpoint, survey architecture (refuses mismatch with config) |
| R1 record | Forward-hook activations at candidate sites over stratified corpus; store fp16 slices + manifest (versioned contract) |
| R2 cleave | Train probes per site × attribute; admit bottlenecks by decodability/selectivity/stability; v1 adds the semvec vector path |
| R3 graft | Insert leaky state banks + modulator bus + FiLM heads (zero-effect at init); core-frozen training (distillation anchor + temporal/priming suites) |
| R4 calibrate | Per-block Δ statistics → delta thresholds at target keep-rate; bounded-divergence validation |
| R5 shims | Train slice-scoped LoRA/steering modules **against recordings only** (core not loaded); admission scoring; semvec-defined shim compilation |
| R6 settle | v0: config passthrough — diffusion-class donors run R0–R5 unchanged; AR→MDLM adaptation is a later separate experiment |
| R7 export | Emit extended GGUF per spec + shim files + registry; `strip` downgrades for stock llama.cpp interop |
| R8 verify | Parity gate, per-feature toggle matrix, behavioral probes, live-runtime shim re-checks |

→ *Details: [arch/runtime/conversion-pipeline.md](../arch/runtime/conversion-pipeline.md)*

## Runtime model (Workstream R)

Capability levels L0–L5 (passthrough → taps/shims → state/modulator → episodic
memory → delta inference → settling decoder → accretion serving) negotiated via
`therobot.features` in the GGUF. Staged roadmap S0–S6, each with a runnable demo
+ regression gate. Verification philosophy: parity is a permanent invariant;
behavioral probes over benchmarks; honest latency (percentiles, never means).

→ *Details: [arch/runtime/README.md](../arch/runtime/README.md) · [llamacpp-extensions.md](../arch/runtime/llamacpp-extensions.md)*

## File formats / data model

No database. State is file-based: donor config YAML → `.lock.yaml` measured
values → recording store (`.npy` + manifest.json) → extended GGUF
(`therobot` / `therobot-shim`) + `registry.json`.

→ *Details: [PROJ-SCHEMA.md](PROJ-SCHEMA.md) · [schema/gguf-format.md](schema/gguf-format.md)*

## Infrastructure

GPU work runs on rented Runpod instances from `runpod/Dockerfile` (CUDA
llama.cpp fork build + `robotgguf` venv with `hf`/`label` extras); corpus,
checkpoints, and pipeline work dirs live on a persistent volume. No cluster
deployment — this is a research toolchain, not a service.

→ *Details: [layout/runpod.md](layout/runpod.md)*

## Technology stack

| Layer | Tech |
|-------|------|
| Conversion pipeline | Python ≥3.9, numpy + pyyaml core; `torch`/`transformers`/`datasets` (`hf` extra); `sentence-transformers`/`fasttext` (`label` extra); `gguf-py` (vendored from the fork) for emission |
| Runtime | llama.cpp deep fork (C/C++, CUDA); new `llama-robot-*` translation units; strict superset of upstream (upstream tests stay green) |
| Teacher labels | Modal-hosted inference endpoint (`convert/tools/modal_teacher.py`) |
| Ground-up research track | Elixir Nx/Axon (planning.md §6 — out of scope for the retrofit path) |

## Key decisions

- **Deep fork over plugin**: single-file extended GGUFs with first-class ops/state; rebase burden accepted and mitigated structurally (fenced `ROBOT-EXT` insertions, upstream-tests-green superset, scheduled syncs).
- **Measured values only**: stages append measurements to the lockfile; the exporter never consumes hand-entered numbers.
- **Versioned contracts**: GGUF spec + recording manifests + semvec spec are never patched in place — breaking changes bump versions (005 §6).
- **Slices, not full activations**: recordings store narrow channel slices (fp16, sharded, memmap) — regenerable derived data, versioned as contracts.
- **Strip for interop**: `robotgguf strip` downgrades extended files to stock-loadable base-arch files; strip-then-compare doubles as the L0 parity fixture.

→ *Full rationale: [arch/runtime/README.md §1–§2](../arch/runtime/README.md) · [arch/proposals/](../arch/proposals/)*
