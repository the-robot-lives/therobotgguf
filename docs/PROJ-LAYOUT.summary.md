# Project Layout — Summary

```
therobotgguf/
├── README.md                       # One-line project statement
├── overview.md                     # Master design overview
├── planning.md                     # Component plan §A–§F
├── notes.md                        # Ground-up design notes
├── extraction-v1.md                # Semvec extraction spec
├── quality-roadmap.md              # Quality/verification roadmap
├── merge-notes.md                  # Branch-sweep bookkeeping
├── arch/                           # Design docs → layout/arch.md
│   ├── runtime/                    # Plan + GGUF extension spec + pipeline/fork specs
│   ├── proposals/                  # 001–006 mechanism proposals
│   └── research/                   # Avenues 1–5 + frontier survey
├── convert/                        # robotgguf pipeline → layout/convert.md
│   ├── robotgguf/                  # Python package (CLI + stage modules)
│   ├── configs/                    # Donor YAMLs + .lock.yaml + semvec-v1.yaml
│   ├── tools/                      # Corpus/site helpers + Modal teacher
│   ├── tests/                      # e2e / extraction / labelers
│   └── docs/                       # Fork-side semvec runtime spec
├── runpod/                         # GPU-host image → layout/runpod.md
│   ├── Dockerfile
│   └── bin/                        # therobot-* helper scripts
├── docs/                           # Developer docs (this tree)
│   ├── PROJ-LAYOUT.md / PROJ-LAYOUT.summary.md
│   ├── PROJ-SCHEMA.md / PROJ-ARCH.md
│   └── layout/                     # Per-directory breakdowns
├── .gitignore                      # llms/, convert/corpus/, convert/work/
└── .git
```
