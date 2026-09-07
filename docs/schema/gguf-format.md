# GGUF File Formats — `therobot` / `therobot-shim`

Extract from `docs/PROJ-SCHEMA.md`. Authoritative source:
`arch/runtime/gguf-extension-spec.md` (draft v1). Rule: contracts are never
patched in place — breaking changes bump `therobot.spec_version`; the runtime
refuses files whose required features or spec version it does not implement.

## Extended model file — arch `therobot`

### Identity & negotiation

| Key | Type | Meaning |
|-----|------|---------|
| `general.architecture` | string | `"therobot"` |
| `therobot.spec_version` | u32 | starts at `1` |
| `therobot.base_architecture` | string | wrapped donor family (`llama`, `qwen2`, `qwen35`, `mamba`, …); all base hparams/tensors use stock keys |
| `therobot.level` | u32 | convenience summary 0–5 (informational) |
| `therobot.features` | [string] | **required** features; loader refuses unknown entries. Values: `taps`, `shims`, `state`, `modulator`, `memory`, `delta`, `settle` |
| `therobot.features_optional` | [string] | ignorable if unknown |
| `therobot.donor.id` | string | HF id + revision of donor |
| `therobot.convert.lockfile_hash` | string | conversion-lockfile hash (provenance) |

Empty `therobot.features` = **L0 passthrough** — must behave identically to
the donor under the base architecture.

### `taps` — bottlenecks + probes

| Key | Type |
|-----|------|
| `therobot.bottleneck.count` | u32 |
| `therobot.bottleneck.{i}.name` | string |
| `therobot.bottleneck.{i}.layer` | u32 (block index) |
| `therobot.bottleneck.{i}.point` | string: `resid_post` \| `attn_out` \| `ffn_out` |
| `therobot.bottleneck.{i}.offset` / `.width` | u32 — channel slice into hidden dim |
| `therobot.bottleneck.{i}.attributes` | [string] |
| `therobot.bottleneck.{i}.decodability` / `.selectivity` | f32 — measured admission scores |

Tensors: `robot.probe.{i}.{attr}.weight` / `.bias` (linear/MLP heads; runtime-
optional to execute).

### `state` — leaky banks

| Key | Type |
|-----|------|
| `therobot.state.bank_count` | u32 |
| `therobot.state.bank.{b}.name` | string: `fast` \| `mid` \| `slow` \| `glacial` |
| `therobot.state.bank.{b}.width` | u32 (channels per covered block) |
| `therobot.state.layers` | [u32] |

Tensors per covered block `L`: `blk.{L}.robot_state.alpha` (per-channel decay
logits), `.in_proj.*`, `.out_proj.*` (**zero at graft** — function-preserving
invariant is checkable from the file). Glacial bank declared as modulator input
(`therobot.modulator.source = "glacial"`).

### `modulator`

| Key | Type |
|-----|------|
| `therobot.modulator.dim` | u32 (8–32) |
| `therobot.modulator.channels` | [string] — e.g. `arousal`, `valence`, `attention` |
| `therobot.modulator.source` | string: `pooled` \| `glacial` |

Tensors: `robot.mod.pool.*`, `robot.mod.cell.*`, `robot.mod.alpha`; per-block
FiLM heads `blk.{L}.robot_film.gamma.*` / `.beta.*` (**identity at graft**:
γ-head bias = 1, all else 0).

### `memory` — episodic store parameters

| Key | Type |
|-----|------|
| `therobot.memory.key_dim` / `.value_dim` | u32 |
| `therobot.memory.capacity` | u32 (runtime may override) |
| `therobot.memory.decay_halflife` | f32 (tokens) |
| `therobot.memory.salience.threshold_quantile` | f32 |

Tensors: `robot.mem.summary.*`, `robot.mem.salience.*`. The store itself is
**session state, not file content**.

### `delta` — change-triggered execution

| Key | Type |
|-----|------|
| `therobot.delta.granularity` | string: `block` (v1) \| `channel_group` (reserved) |
| `therobot.delta.heartbeat` | u32 — tokens between dense sweeps |
| `therobot.delta.target_keep_rate` | f32 — calibration provenance |

Tensors: `blk.{L}.robot_delta.theta_base`, `blk.{L}.robot_delta.fatigue.*`,
`robot.delta.excitability.*`.

### `settle` — settling decoder

| Key | Type |
|-----|------|
| `therobot.settle.objective` | string: `mdlm` \| … |
| `therobot.settle.mask_token_id` | u32 |
| `therobot.settle.max_steps` / `.epsilon` | u32 / f32 |
| `therobot.settle.m_schedule` | [f32] — m-arousal → extra rounds |

Tensors: `robot.settle.len.*`, objective-specific heads.

## Quantization

Base tensors quantize normally (fork's `llama-quantize`). **All `robot.*` /
`blk.*.robot_*` tensors are exempt** (stay f16/f32). Probe/FiLM heads must
survive base quantization unchanged — re-verified post-quant by R8.

## Interop — `robotgguf strip`

Extended files do **not** load in stock llama.cpp (deep-fork decision). `strip`
drops all extension tensors/keys and rewrites `general.architecture` to
`therobot.base_architecture` → stock-loadable file. Strip-then-compare doubles
as the L0 parity test fixture.

## Shim module file — arch `therobot-shim`

Shims ship as standalone hot-loadable GGUFs, never baked into the model file.

| Key | Type |
|-----|------|
| `general.architecture` | `"therobot-shim"` |
| `therobot.spec_version` | u32 |
| `therobot.shim.name` / `.version` | string |
| `therobot.shim.target_model` | string — donor/model hash scores were measured against |
| `therobot.shim.target_bottleneck` | string — bottleneck `name` |
| `therobot.shim.effect` | string — human-readable verified effect |
| `therobot.shim.selectivity` | f32 — admission score |
| `therobot.shim.depends` / `.conflicts` | [string] — other shim names (honored at load) |
| `therobot.shim.gate` | string: `always` \| `modulator:<channel><op><val>` \| `probe:<attr><op><val>` |

Tensors: `robot.shim.a.*` / `.b.*` (LoRA-style low-rank pair on the slice),
`.steer.*` (additive), `.gain.*` (multiplicative), `.gate.*` (optional gating
MLP). A registry index **`registry.json`** (converter tooling) lists admitted
shims per model hash with scores.

## Reserved

Lifecycle metadata (006: unit age/vitality, spawn provenance) is intentionally
**not** in v1 — exports are always recompacted static graphs. Would land as
`therobot.lifecycle.*` in a spec bump if ever needed.
