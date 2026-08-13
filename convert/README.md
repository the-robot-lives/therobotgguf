# robotgguf — the conversion pipeline (workstream C)

Retrofits existing LLM checkpoints onto the therobot runtime
([`../arch/runtime/conversion-pipeline.md`](../arch/runtime/conversion-pipeline.md)).
The prime directive holds end-to-end: the donor core is frozen from the moment
of recording, every graft is function-preserving at insertion, and the
pipeline's own verify stage proves it against the live runtime — a converted
model with all extensions at init is **bit-for-bit the donor** (checked, not
asserted).

## Usage

The current primary target is **Qwen3.5-0.8B** (`configs/qwen3.5-0.8b.yaml`,
with donor-specific instructions in `configs/qwen3.5-0.8b.md`); the
`qwen2.5-1.5b.yaml` config remains as the reference for a pure-attention
donor. Substitute your config below.

```bash
cd projects/therobotgguf/convert
pip install -e .            # numpy + pyyaml; `pip install -e '.[hf]'` for R0/R1/R3 training
robotgguf --config configs/qwen2.5-1.5b.yaml ingest      # R0  [HF stack]
robotgguf --config configs/qwen2.5-1.5b.yaml record      # R1  [HF stack]
robotgguf --config configs/qwen2.5-1.5b.yaml relabel     # R1.5 labels from stored tokens (semvec t0 + views)
robotgguf --config configs/qwen2.5-1.5b.yaml labelvec --tiers t1   # R1.5 t1/t2 layering [label stack]
robotgguf --config configs/qwen2.5-1.5b.yaml labels-qa   # C2 label quality gates
robotgguf --config configs/qwen2.5-1.5b.yaml cleave      # R2  (v0 categorical + semvec vector paths)
robotgguf --config configs/qwen2.5-1.5b.yaml graft       # R3  (--steps N trains; 0 = zero-init)
robotgguf --config configs/qwen2.5-1.5b.yaml calibrate   # R4
robotgguf --config configs/qwen2.5-1.5b.yaml shims       # R5
robotgguf --config configs/qwen2.5-1.5b.yaml shim-compile # semvec-defined shims (extraction-v1 §4.5)
robotgguf --config configs/qwen2.5-1.5b.yaml settle      # R6  (config only, v0)
robotgguf --config configs/qwen2.5-1.5b.yaml export --out work/qwen-therobot.gguf   # R7
robotgguf --config configs/qwen2.5-1.5b.yaml verify --parity-bin /tmp/robot_parity_test  # R8
```

Stages append *measured* values to `<config>.lock.yaml`; the exporter consumes
only measured values, never hand-entered ones. `robotgguf strip <in> <out>`
downgrades an extended file for stock-llama.cpp interop.

## Status

| Stage | State |
|---|---|
| R0 ingest, R1 record, R3 graft training | implemented, **untested** — need a GPU/checkpoint host (`pip install '.[hf]'`); the graft's zero-init path runs anywhere and is tested |
| R2 cleave, R4 calibrate, R5 shims, R7 export/strip, R8 verify | implemented and covered by `tests/e2e_test.py` |
| R6 settle | config passthrough (v0 policy: diffusion-class donor through R0–R5; the runtime's `jacobi-ar` objective works on any causal donor) |
| Weak labelers (R2 label source) | heuristic v0 in `robotgguf/labelers.py` — sentence-granular, seven attributes, wired into `record`; `robotgguf relabel` regenerates labels from stored tokens without re-running the model; a teacher-LLM pass can overwrite `labels/<attr>.npy` later (same contract). Tested by `tests/labelers_test.py` |
| **extraction-v1** (C1/C2/C4 core) | implemented — see `../extraction-v1.md`. C1: corpus manifest + stratified loader (`robotgguf/corpus.py`, fixes the v0 head-of-file sampling bug) + `tools/fetch_corpus.py` seven-stratum fetcher. C2: semvec/v1 spec (`configs/semvec-v1.yaml`), label vector + tiered labelers (`semvec.py`, `labelers_t0/t1/teacher.py`), `robotgguf labelvec` (t1/t2 layering, `--fit-basis`). C4: vector cleave (`cleave_vec.py`) — ridge map per site, per-axis admission with domain stability, MLP fallback findings, and the readout/overlay pair (`{site}.proj/.calib/.overlay.npy`, write-calibrated — `overlay.py`). R7 now packages the readout layer (`robot.semvec.*.proj/.calib/.overlay` + `therobot.semvec.*` KVs, optional-feature flagged; `strip` removes them), and `robotgguf shim-compile` compiles semvec-defined shims per donor through the write-calibrated overlay with crosstalk-based admission (`semvec_shims:` in the config). `robotgguf verify` gate 5 checks the packaged readout layer structurally; `robotgguf labels-qa` reports label quality; `tools/gen_sites.py` generates site grids. The fork-side contract is `docs/semvec-runtime-spec.md`. Tested by `tests/extraction_test.py` (no GPU needed). GPU passes (C3/C5 record, t1/t2 labels, R3 training) run on a rented GPU host via the plain CLI; the teacher endpoint is Modal (`tools/modal_teacher.py`, inference only). Donor configs: `configs/qwen3.6-35b-a3b.yaml` (primary, verified vs HF config.json) alongside qwen3.5-0.8b |

## End-to-end test (no GPU needed)

Runs synthetic recordings (a legitimate stand-in — recordings are the
versioned contract) through cleave → graft(init) → calibrate → shims →
export → verify against the fork's tiny fixture donor:

```bash
python3 tests/e2e_test.py /tmp/robot-fixtures <fork-root> /tmp/robot-build/bin /tmp/robot_parity_test
# expect: CONVERSION E2E: OK — including a bit-exact parity gate on the export
```

Requires the fork built (`llama-robot-inspect`) and the fixture set from
`<fork>/tests/robot/make_donor_gguf.py`.
