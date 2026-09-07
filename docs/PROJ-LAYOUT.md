# Project Layout — therobotgguf

Docs-heavy research project: retrofit existing LLM checkpoints onto the "therobot"
runtime via a deep fork of llama.cpp. Three top-level concerns: **design docs**
(root `*.md` + `arch/`), the **Python conversion pipeline** (`convert/`), and
**GPU-host packaging** (`runpod/`). There is no application server and no database;
see `PROJ-SCHEMA.md` for the file formats it *does* own.

```
therobotgguf/
├── README.md                       # One-line project statement
├── overview.md                     # Master design overview (large; start here for context)
├── planning.md                     # Component plan §A–§F referenced by arch/
├── notes.md                        # Ground-up design notes (noizu ai notes)
├── extraction-v1.md                # extraction-v1 spec (semvec C1/C2/C4 core)
├── quality-roadmap.md              # Quality/verification roadmap
├── merge-notes.md                  # Branch-sweep / merge bookkeeping (dated)
├── arch/                           # Runtime & conversion design → [layout/arch.md](layout/arch.md)
│   ├── runtime/                    # Conversion pipeline, llama.cpp fork plan, GGUF extension spec
│   ├── proposals/                  # Proposals 001–006 (learning, state, delta, settle, accretion)
│   └── research/                   # Research avenues 1–5 + frontier survey
├── convert/                        # `robotgguf` conversion pipeline → [layout/convert.md](layout/convert.md)
│   ├── robotgguf/                  # Python package — CLI + one module per pipeline stage
│   ├── configs/                    # Per-donor YAML configs + measured `.lock.yaml` files
│   ├── tools/                      # Corpus fetch/gen + Modal teacher endpoint
│   ├── tests/                      # e2e / extraction / labelers test suites (no GPU)
│   └── docs/                       # Fork-side runtime contract (semvec-runtime-spec.md)
├── runpod/                         # Rented-GPU-host image → [layout/runpod.md](layout/runpod.md)
│   ├── Dockerfile                  # CUDA llama.cpp fork build + robotgguf venv
│   └── bin/                        # therobot-* volume/corpus/model helper scripts
├── .gitignore                      # Ignores llms/, convert/corpus/, convert/work/
└── .git                            # Submodule pointer (monorepo: Portfolio/Apps/AI/therobotgguf)
```

## Generated / runtime directories (gitignored — never document contents as source)

| Path | Created by | Contents |
|------|-----------|----------|
| `convert/work/` | pipeline stages | recordings, lockfile side-effects, exported GGUFs, per-stage outputs |
| `convert/corpus/` | `tools/fetch_corpus.py` / `make_corpus.py` | downloaded / generated training text |
| `llms/` | `runpod/bin/therobot-fetch-model` | donor checkpoints (HF or GGUF) |
| `convert/robotgguf.egg-info/` | `pip install -e .` | packaging metadata |

## Key files requiring setup

| File | Action |
|------|--------|
| `convert/configs/qwen3.5-0.8b.yaml` | Primary donor config — check `donor:` path and `paths:` (gguf-py, fork `runtime_bin`) point at your local checkout |
| `convert/configs/*.lock.yaml` | Do NOT hand-edit — stages append measured values; exporter consumes only these |
| `convert/configs/semvec-v1.yaml` | Versioned standard (append-only axes) — never edit in place; bump version |
| fork checkout | Pipeline expects the llama.cpp fork at `paths.gguf_py` / `paths.runtime_bin` (see `convert/configs/qwen3.5-0.8b.md`) |

## Entry points

| Entry | What |
|-------|------|
| `robotgguf` CLI (`convert/robotgguf/cli.py`) | All pipeline stages: ingest → record → relabel → labelvec → labels-qa → cleave → graft → calibrate → shims → shim-compile → settle → export → verify, plus `strip` |
| `python3 convert/tests/e2e_test.py` | Full no-GPU conversion e2e incl. bit-exact parity gate |
| `runpod/bin/therobot-*` | GPU-host volume init/verify, corpus + model fetch |
