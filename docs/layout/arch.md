# arch/ — Runtime & Conversion Design Docs

Status: plan / pre-implementation designs. Rooted in `../../planning.md` §A–§F and
`../../overview.md`. The runtime itself lives in a **separate llama.cpp fork** — this
repo holds the specs and the converter that targets it.

```
arch/
├── README.md                       # (none at this level — see runtime/README.md for the plan)
├── runtime/                        # The staged plan + the contracts both workstreams implement
│   ├── README.md                   # Start here — decisions, L0–L5 levels, S0–S6 roadmap, verification philosophy
│   ├── conversion-pipeline.md      # Workstream C: R0–R8 stage spec (ingest → … → verify)
│   ├── llamacpp-extensions.md      # Workstream R: llama.cpp deep-fork plan (E0–E8), rebase mitigation
│   └── gguf-extension-spec.md      # THE contract: `therobot` GGUF key/tensor spec, v1 (versioned, never patched)
├── proposals/                      # Numbered mechanism proposals (001–006)
│   ├── README.md                   # Index + cross-proposal synergy notes
│   ├── 001-sundered-backprop-local-learning.md
│   ├── 002-event-driven-delta-inference.md
│   ├── 003-multi-timescale-state-backbone.md
│   ├── 004-settle-to-answer-parallel-generation.md
│   ├── 005-frozen-core-accretion.md
│   └── 006-unit-lifecycle-dormancy-and-spawning.md
└── research/                       # Exploratory research avenues
    ├── README.md
    ├── frontier-survey-2026-07.md
    ├── avenue-1-jspace-injection.md
    ├── avenue-2-metamorphosis.md
    ├── avenue-3-resolution-stepping.md
    ├── avenue-4-latent-depth-continuity.md
    └── avenue-5-mathematical-instruments.md
```

## Reading order

1. `runtime/README.md` — how workstreams C (conversion) and R (runtime) interlock
2. `runtime/gguf-extension-spec.md` — the file-format contract (see also `PROJ-SCHEMA.md`)
3. `../../extraction-v1.md` — the semvec extraction standard the converter implements
4. Proposals as referenced by the roadmap stage you're working on
