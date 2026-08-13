# semvec runtime extension — spec for the fork (handoff)

**Status:** BOTH sides implemented. Conversion: `robotgguf` emits everything
below (`tests/extraction_test.py` round-trips it via gguf-py). Fork: the
`semvec` optional feature is implemented per this spec —
`src/llama-robot-semvec.{h,cpp}` (validation, read/query, overlay-as-
ephemeral-E3-shim), hparams negotiation + site→tap resolution, `qwen35moe`
factory registration, `tests/robot/robot_semvec_test.cpp` (the §6 gates) —
**pending first build + fixture run on the Mac** (authored off-box; expect
only mechanical compile fixes if any). Companion: `../../extraction-v1.md`.

---

## 1. What the runtime gains

The **standardized readout layer**: per admitted site, a projection of the
donor's residual-stream slice into semvec (a versioned, donor-independent
512-dim semantic coordinate system), plus the calibrated write path back.
Three capabilities, all optional-feature gated:

1. **Read** — the model's current state in standard coordinates, on demand.
2. **Zero-shot query** — any text question answered against that state
   (host-side; the runtime only supplies the vector).
3. **Overlay** — a semvec-space edit Δs applied onto the stream
   (`h' = h + Δs·G`), the portable-shim write path. Δs = 0 must be
   bit-exact identity.

## 2. GGUF contract (emitted by R7 today)

KVs — all under the negotiated file, absent on stock files:

```
therobot.features_optional        [str]   contains "semvec" (optional form)
                                          — OR "semvec" ∈ therobot.features
                                          (required form; loader refuses if
                                          unimplemented, per E1 rules)
therobot.semvec.version           str     e.g. "1.0"
therobot.semvec.hash              str     16-hex coordinate-system identity
therobot.semvec.named_dim         u32     128
therobot.semvec.latent_dim        u32     384
therobot.semvec.axes              [str]   named-axis names, length = named_dim
therobot.semvec.site_count        u32
therobot.semvec.site.{i}.name     str     e.g. "resid18"
therobot.semvec.site.{i}.layer    u32
therobot.semvec.site.{i}.point    str     resid_post | attn_out | ffn_out
therobot.semvec.site.{i}.offset   u32     channel offset of the slice
therobot.semvec.site.{i}.width    u32     slice width d
therobot.semvec.site.{i}.n_admitted  u32
therobot.semvec.site.{i}.n_writable  u32
```

Tensors (F32):

```
robot.semvec.{site}.proj      [d, D]   E — read map (unadmitted columns zero)
robot.semvec.{site}.calib     [D, 2]   per-axis (scale, bias); scale ∈ {0,1}
robot.semvec.{site}.overlay   [D, d]   G — write map (unwritable rows zero;
                                       admitted rows write-calibrated so
                                       G·E diagonal == 1)
```

D = named_dim + latent_dim. Semantics: `s = h_slice·E + calib[:,1]`, valid
only where `calib[:,0] == 1`. Loader claims these tensors during load (same
bookkeeping as `robot.probe.*`).

Shim modules may carry provenance KVs (`therobot.shim.semvec.version/hash/
axis/scale`) — informational; the E3 attach path treats the module's
`robot.shim.steer` exactly as today.

## 3. Negotiation rules (E1)

- `"semvec"` in `therobot.features_optional`: load the tensors if the
  feature is implemented, silently ignore otherwise. Older runtimes must
  keep loading the file (this is the shipping default).
- `"semvec"` in `therobot.features` (required): refuse if unimplemented —
  standard E1 behavior, nothing new.
- If implemented: refuse a file whose semvec tensor shapes disagree with the
  KV dims, and refuse `semvec_query`/module attach when the caller's semvec
  hash ≠ the file's (coordinate systems must match).

## 4. Public API (`include/llama-robot.h` additions)

```c
// number of semvec sites in the loaded model (0 → feature absent)
int32_t llama_robot_semvec_site_count(const struct llama_model *model);
const char *llama_robot_semvec_site_name(const struct llama_model *model, int32_t i);

// read the current standardized state at site i (most recent position).
// dst must hold D floats (named_dim + latent_dim). Applies proj + calib;
// non-admitted axes are written as 0. One matvec, computed only when called
// (same cost posture as llama_robot_probe_eval).
int32_t llama_robot_semvec_read(struct llama_context *ctx, int32_t site,
                                float *dst, size_t n);

// calibrated named-axis convenience: value of axis (by index into
// therobot.semvec.axes) from a previously read vector — pure host math.
float llama_robot_semvec_axis(const struct llama_model *model,
                              const float *s, int32_t axis);

// zero-shot: cosine between a read vector and a caller-supplied query
// vector (the host embeds + reduces the query text through the frozen
// semvec basis — the runtime never sees the embedder). Latent block only
// by default; full-vector variant behind a flag.
float llama_robot_semvec_query(const struct llama_model *model,
                               const float *s, const float *query, size_t n);

// overlay: register a semvec-space edit at site i for subsequent decodes:
// h_slice' = h_slice + scale · (delta_s · G). delta_s == NULL or scale == 0
// detaches. Gated + epoch-keyed exactly like E3 shim attach/detach.
int32_t llama_robot_semvec_overlay_set(struct llama_context *ctx, int32_t site,
                                       const float *delta_s, float scale);
```

## 5. Graph integration

- **Read path:** reuse the E2 tap machinery verbatim — each semvec site is a
  tap (`ggml_view` + `ggml_cont`, first-class graph output). `semvec_read`
  is `tap_read` followed by one host-side matvec against proj (D×d ≈ 512×256
  ≈ 128k MACs — negligible). No graph change at all; sites whose slice
  coincides with a declared tap share the node.
- **Write path:** `Δs·G` is a *constant* d-vector for a given overlay_set
  call — precompute it host-side at set time, upload as a graph input tensor
  (set-input time, like modulator state), and splice `h' = h + steer_input`
  into the slice with the SAME rewiring machinery E3 uses (expand node,
  re-point consumers, restore order, fix use-counts, epoch-keyed reuse).
  This makes the overlay op literally an E3 shim whose steering vector is
  runtime-settable — no new splice code, only a new input tensor.
- **Parity:** overlay unset (or Δs = 0) contributes nothing to the graph
  (input tensor zero → add is identity; or skip the splice entirely when no
  overlay was ever set — preferred, keeps L0 files byte-identical in graph
  topology too).

## 6. Gates (R8 additions — permanent CI)

1. **Dormant-semvec parity:** a file carrying semvec tensors with no
   overlay set reports `max |logit diff| = 0` vs its stripped twin (extends
   the existing parity gate; the conversion side's `verify` already checks
   the *structural* half — calib scales, zeroed columns, G·E diagonal — via
   `robotgguf verify` gate 5).
2. **Read determinism:** `semvec_read` twice at the same position returns
   identical vectors; taps-active parity already covers non-perturbation.
3. **Write roundtrip:** set overlay `Δs = e_j` (unit write on a writable
   axis), decode one token, `semvec_read` at the same site moves axis j by
   1.0 ± 1e-3 and no unwritable axis moves.
4. **Zero-shot sanity (needs the frozen basis):** the host-side suite in
   `robotgguf verify` scores subject/tone/register contrasts on held-out
   text; pass bar recorded in the lockfile.

## 7. Also on the fork's plate

- `qwen35moe` arch registration wrapping the qwen3_5_moe family (Qwen3.6
  MoE: 256 experts / 8 active, shared expert, MTP head to be ignored at
  conversion like stock does; text stack nests under the multimodal
  wrapper). The template-wrapper design (E1) means hparams + tensor mapping,
  no new therobot code.
- `llama-robot-inspect`: print the semvec block (version/hash/dims/sites/
  admitted/writable counts) from the manifest.

## 8. Fence budget

Expected new fences: 0 in upstream files if the tap + E3-splice reuse holds
(all new code in `llama-robot-semvec.{h,cpp}` + dispatch from
`llama_robot_graph_apply`). Anything beyond that must be enumerated in
`docs/robot/patch-points.md` per fork hygiene.
