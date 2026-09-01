# convert/ — `robotgguf` Conversion Pipeline (Workstream C)

Python package implementing `arch/runtime/conversion-pipeline.md` stages R0–R8.
Prime directive: donor core frozen from recording; every graft function-preserving
at insertion; `verify` proves bit-for-bit parity. Install: `pip install -e .`
(add `.[hf]` / `.[label]` for GPU/teacher stages).

```
convert/
├── README.md                       # Stage usage, status matrix, e2e instructions
├── pyproject.toml                  # Package def; deps numpy+pyyaml, extras hf/label; entry point `robotgguf`
├── uv.lock                         # Locked dep resolution (uv)
├── robotgguf/                      # The package — CLI + one module per concern
│   ├── cli.py                      # Entry point: stage subcommands + --config
│   ├── config.py                   # Config dataclass + lockfile read/append (`<config>.lock.yaml`)
│   ├── ingest.py                   # R0 — load HF checkpoint, survey arch (refuses mismatch)
│   ├── record.py                   # R1 — capture bottleneck activations over corpus windows
│   ├── recordings.py               # Recording store format (versioned contract)
│   ├── corpus.py                   # C1 — corpus manifest + stratified interleaved loader
│   ├── labelers.py                 # R2 weak labeler v0 (heuristic, 7 attributes)
│   ├── labelers_t0.py              # t0 structural/heuristic labels (semvec tiers)
│   ├── labelers_t1.py              # t1 classifier/embedder labels
│   ├── labelers_teacher.py         # t2 teacher-LLM label pass (Modal endpoint)
│   ├── semvec.py                   # C2 — semvec/v1 label vector + readout coordinates
│   ├── cleave.py                   # R2 — v0 categorical path: train probes, admit bottlenecks
│   ├── cleave_vec.py               # C4 — vector cleave: ridge maps, per-axis admission, MLP fallback
│   ├── overlay.py                  # C4 readout/overlay pair (.proj/.calib/.overlay.npy, write-calibrated)
│   ├── graft.py                    # R3 — state banks + modulator graft (zero-init = function-preserving)
│   ├── calibrate.py                # R4 — delta thresholds from measured keep rates
│   ├── shims.py                    # R5 — shim definition/verification (v0 set)
│   ├── shimc.py                    # semvec-defined shim compiler (extraction-v1 §4.5)
│   ├── settle.py                   # R6 — settle config passthrough (v0)
│   ├── export.py                   # R7 — extended-GGUF writer + `strip` downgrade
│   ├── verify.py                   # R8 — parity/structural gates via the fork's llama-robot-inspect
│   └── qa.py                       # C2 label quality gates (`labels-qa`)
├── configs/                        # Per-donor declarative configs + measured lockfiles
│   ├── qwen3.5-0.8b.yaml           # Primary donor config (heavily commented — the reference)
│   ├── qwen3.5-0.8b.md             # Donor-specific runbook (fork convert step, etc.)
│   ├── qwen3.5-0.8b-init.yaml / .lock.yaml  # Stage inputs / appended measured outputs
│   ├── qwen2.5-1.5b.yaml           # Pure-attention reference donor
│   ├── qwen3.5-9b.yaml, qwen3.6-35b-a3b.yaml  # Larger donors (35b verified vs HF config.json)
│   └── semvec-v1.yaml              # VERSIONED STANDARD — semvec axis spec (append-only)
├── tools/                          # Offline helpers
│   ├── fetch_corpus.py             # Seven-stratum corpus fetcher
│   ├── make_corpus.py              # Corpus assembly
│   ├── gen_sites.py                # Cleave-site grid generator
│   └── modal_teacher.py            # Teacher-LLM endpoint on Modal (inference only)
├── tests/                          # No-GPU test suites
│   ├── e2e_test.py                 # Full conversion e2e incl. bit-exact parity gate
│   ├── extraction_test.py          # semvec/readout layer tests
│   ├── labelers_test.py            # Weak labeler tests
│   └── make_corpus.py              # Test fixture corpus helper
└── docs/
    └── semvec-runtime-spec.md      # Fork-side contract for the packaged readout layer
```

## Pipeline order (see README.md for full commands)

ingest → record → relabel → labelvec → labels-qa → cleave → graft → calibrate →
shims → shim-compile → settle → export → verify. Stage outputs append to the
`.lock.yaml`; the exporter consumes only measured values.
