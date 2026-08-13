# GGUF Extension Spec — `therobot` v1 (draft)

**Status:** draft spec, versioned; the single contract between the conversion pipeline and the runtime.
**Rule inherited from 005 §6:** contracts are never patched in place. Any breaking change bumps `therobot.spec_version`; the runtime refuses files whose *required* features or spec version it does not implement.

---

## 1. Model files

### 1.1 Identity & negotiation

| Key | Type | Meaning |
|---|---|---|
| `general.architecture` | string | `"therobot"` |
| `therobot.spec_version` | u32 | starts at `1` |
| `therobot.base_architecture` | string | wrapped donor family: `"llama"`, `"qwen2"`, `"mamba"`, … All base hparams/tensors use that family's stock keys/names unchanged |
| `therobot.level` | u32 | convenience summary (0–5), informational only |
| `therobot.features` | [string] | required features; loader must refuse unknown entries. Values: `taps`, `shims`, `state`, `modulator`, `memory`, `delta`, `settle` |
| `therobot.features_optional` | [string] | ignorable if unknown |
| `therobot.donor.id` | string | HF id + revision of the donor |
| `therobot.convert.lockfile_hash` | string | hash of the conversion lockfile that produced this file (provenance) |

A file with an empty `therobot.features` list is an **L0 passthrough** and must behave identically to the donor under the base architecture.

### 1.2 Bottlenecks / taps (`taps`)

| Key | Type |
|---|---|
| `therobot.bottleneck.count` | u32 |
| `therobot.bottleneck.{i}.name` | string |
| `therobot.bottleneck.{i}.layer` | u32 (block index) |
| `therobot.bottleneck.{i}.point` | string: `resid_post` \| `attn_out` \| `ffn_out` |
| `therobot.bottleneck.{i}.offset` / `.width` | u32 — channel slice into the hidden dim |
| `therobot.bottleneck.{i}.attributes` | [string] |
| `therobot.bottleneck.{i}.decodability` / `.selectivity` | f32 — admission scores measured at convert time |

Probe tensors: `robot.probe.{i}.{attr}.weight` / `.bias` (small linear/MLP heads; runtime-optional to execute).

### 1.3 Leaky state (`state`)

| Key | Type |
|---|---|
| `therobot.state.bank_count` | u32 |
| `therobot.state.bank.{b}.name` | string: `fast` \| `mid` \| `slow` \| `glacial` |
| `therobot.state.bank.{b}.width` | u32 (channels per covered block) |
| `therobot.state.layers` | [u32] — blocks carrying state branches |

Per covered block `L`: `blk.{L}.robot_state.alpha` (per-channel decay logits), `blk.{L}.robot_state.in_proj.*`, `blk.{L}.robot_state.out_proj.*` (zero at graft — the function-preserving invariant is checkable from the file). Glacial-bank slices are declared as the modulator's input source (`therobot.modulator.source = "glacial"`).

### 1.4 Modulator (`modulator`)

| Key | Type |
|---|---|
| `therobot.modulator.dim` | u32 (8–32) |
| `therobot.modulator.channels` | [string] — named, e.g. `arousal`, `valence`, `attention` |
| `therobot.modulator.source` | string: `pooled` \| `glacial` |

Tensors: `robot.mod.pool.*`, `robot.mod.cell.*` (update GRU/MLP), `robot.mod.alpha` (per-channel decay), and per-block FiLM heads `blk.{L}.robot_film.gamma.*` / `.beta.*` (identity at graft: γ-head bias = 1, all else 0).

### 1.5 Memory (`memory`)

| Key | Type |
|---|---|
| `therobot.memory.key_dim` / `.value_dim` | u32 |
| `therobot.memory.capacity` | u32 (runtime may override) |
| `therobot.memory.decay_halflife` | f32 (tokens) |
| `therobot.memory.salience.threshold_quantile` | f32 |

Tensors: `robot.mem.summary.*` (bottleneck→key/value projections), `robot.mem.salience.*` (gate parameters). The store itself is session state, not file content.

### 1.6 Delta inference (`delta`)

| Key | Type |
|---|---|
| `therobot.delta.granularity` | string: `block` (v1) \| `channel_group` (reserved) |
| `therobot.delta.heartbeat` | u32 — tokens between dense sweeps |
| `therobot.delta.target_keep_rate` | f32 — calibration provenance |

Tensors: `blk.{L}.robot_delta.theta_base`, `blk.{L}.robot_delta.fatigue.*`, `robot.delta.excitability.*` (m → θ offsets).

### 1.7 Settling (`settle`)

| Key | Type |
|---|---|
| `therobot.settle.objective` | string: `mdlm` \| … |
| `therobot.settle.mask_token_id` | u32 |
| `therobot.settle.max_steps` / `.epsilon` | u32 / f32 |
| `therobot.settle.m_schedule` | [f32] — map from m-arousal to extra rounds |

Tensors: `robot.settle.len.*` (length head), any objective-specific heads.

## 2. Quantization

Base tensors quantize normally (`llama-quantize` in the fork). All `robot.*` / `blk.*.robot_*` tensors are exempt from quantization (small; stay f16/f32). Probe and FiLM heads must survive quantization of the base unchanged — admission scores are re-verified post-quant by R8.

## 3. Interop

Extended files do **not** load in stock llama.cpp (deep-fork decision). `robotgguf strip <in> <out>` drops all extension tensors/keys and rewrites `general.architecture` to `therobot.base_architecture`, producing a stock-loadable file. Strip-then-compare is also the L0 parity test fixture.

## 4. Shim module files (`therobot-shim`)

Shims ship as standalone GGUFs (hot-loadable, registry-managed — the 005 module economy), never baked into the model file:

| Key | Type |
|---|---|
| `general.architecture` | `"therobot-shim"` |
| `therobot.spec_version` | u32 |
| `therobot.shim.name` / `.version` | string |
| `therobot.shim.target_model` | string — donor/model hash the admission scores were measured against |
| `therobot.shim.target_bottleneck` | string — bottleneck `name` (§1.2) |
| `therobot.shim.effect` | string — human-readable verified effect |
| `therobot.shim.selectivity` | f32 — admission score |
| `therobot.shim.depends` / `.conflicts` | [string] — other shim names |
| `therobot.shim.gate` | string: `always` \| `modulator:<channel><op><val>` \| `probe:<attr><op><val>` |

Tensors: `robot.shim.a.*` / `robot.shim.b.*` (LoRA-style low-rank pair on the slice), `robot.shim.steer.*` (additive), `robot.shim.gain.*` (multiplicative), `robot.shim.gate.*` (optional gating MLP). A registry index (`registry.json`, maintained by converter tooling) lists admitted shims per model hash with their scores — the runtime honors `depends`/`conflicts` at load.

## 5. Reserved for the ground-up track

Lifecycle metadata (006: unit age/vitality, spawn provenance) is intentionally **not** in v1 — exports are always recompacted static graphs (006 §8). If lifecycle-annotated exports ever matter (e.g. Accord-style archival of retired units), that lands as `therobot.lifecycle.*` in a spec-version bump.
