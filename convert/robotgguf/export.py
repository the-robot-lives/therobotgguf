"""R7 — Export: emit the extended GGUF per gguf-extension-spec.md, and
`strip` (extended → stock) for interop.

The base file is a stock GGUF of the donor (from the fork's
convert_hf_to_gguf.py, or config.base_gguf). Export copies every base KV and
tensor untouched, rewrites general.architecture to "therobot", writes the
therobot.* contract from the lockfile's *measured* sections, and appends the
extension tensors (probes from R2, grafts from R3, thresholds from R4).
Zero-effect grafts keep the file bit-for-bit the donor in behavior — R8
checks that against the live runtime.
"""
from __future__ import annotations

import os

import numpy as np

from . import SPEC_VERSION
from .config import Config, Lockfile


def _copy_base(gguf, reader, writer) -> None:
    """Copy every KV (except the architecture) and every tensor, unchanged —
    the pattern gguf-py's own gguf_new_metadata uses."""
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)
    for tensor in reader.tensors:
        writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype,
                               tensor.data.nbytes, tensor.tensor_type)


def _write_tensor_data(reader, writer, ext=()) -> None:
    """Write header/KVs/tensor-infos, then the data — base tensors in reader
    order followed by the extension arrays, matching the info order."""
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for tensor in reader.tensors:
        writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)
    for _, data in ext:
        writer.write_tensor_data(data)
    writer.close()


def run(cfg: Config, out_path: str) -> None:
    gguf = cfg.gguf_module()
    lock = Lockfile(cfg.lockfile_path)

    if not cfg.base_gguf:
        raise SystemExit("config.base_gguf is required (convert the donor with the "
                         "fork's convert_hf_to_gguf.py first)")
    base = cfg.resolve(cfg.base_gguf)
    reader = gguf.GGUFReader(base)

    arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
    if arch != cfg.base_architecture:
        raise SystemExit(f"base GGUF architecture '{arch}' != config base_architecture "
                         f"'{cfg.base_architecture}'")

    writer = gguf.GGUFWriter(out_path, "therobot")
    _copy_base(gguf, reader, writer)

    # §1.1 identity & negotiation — measured values only
    features = list(cfg.features)
    writer.add_uint32("therobot.spec_version", SPEC_VERSION)
    writer.add_string("therobot.base_architecture", cfg.base_architecture)
    writer.add_uint32("therobot.level", int(cfg.level))
    writer.add_array("therobot.features", features)
    writer.add_string("therobot.donor.id", cfg.donor)
    writer.add_string("therobot.convert.lockfile_hash", cfg.config_hash())

    ext = []  # (name, ndarray)

    # §1.2 bottlenecks + probe tensors (R2)
    if "taps" in features:
        cleaved = lock.require("cleave", "cleave")
        bns = cleaved["bottlenecks"]
        probe_dir = cfg.resolve(cleaved["probe_dir"])
        writer.add_uint32("therobot.bottleneck.count", len(bns))
        for i, bn in enumerate(bns):
            p = f"therobot.bottleneck.{i}."
            writer.add_string(p + "name", bn["name"])
            writer.add_uint32(p + "layer", int(bn["layer"]))
            writer.add_string(p + "point", bn["point"])
            writer.add_uint32(p + "offset", int(bn["offset"]))
            writer.add_uint32(p + "width", int(bn["width"]))
            writer.add_array(p + "attributes", list(bn["attributes"]))
            writer.add_float32(p + "decodability", float(bn["decodability"]))
            writer.add_float32(p + "selectivity", float(bn["selectivity"]))
            for attr in bn["attributes"]:
                for part in ("weight", "bias"):
                    f = os.path.join(probe_dir, f"{bn['name']}.{attr}.{part}.npy")
                    if os.path.exists(f):
                        ext.append((f"robot.probe.{i}.{attr}.{part}",
                                    np.load(f).astype(np.float32)))

    # §1.2b the standardized readout layer (extraction-v1 §4.4) — proj/calib/
    # overlay per admitted site, emitted whenever cleave_vec has run. Shipped
    # as an OPTIONAL feature (therobot.features_optional): pure additional
    # outputs + a dormant write path, so older runtimes ignore it and parity
    # is untouched. Listing "semvec" in cfg.features instead makes it
    # REQUIRED (a file that must not load without its readout layer).
    cvec = lock.section("cleave_vec")
    if cvec:
        if cfg.semvec:
            from .semvec import SemvecSpec  # noqa: PLC0415
            spec = SemvecSpec.load(cfg.resolve(cfg.semvec))
            if cvec["semvec"]["hash"] != spec.spec_hash():
                raise SystemExit("export: lockfile cleave_vec semvec hash "
                                 f"{cvec['semvec']['hash']} != spec {spec.spec_hash()}")
            axis_names = [a.name for a in spec.axes]
            named_dim, latent_dim = spec.named_dim, spec.latent_dim
        else:
            axis_names, named_dim, latent_dim = [], 0, 0

        vdir = cfg.resolve(cvec["probe_dir"])
        sv_sites = []
        for sname, info in cvec["sites"].items():
            pf = os.path.join(vdir, f"{sname}.proj.npy")
            if not os.path.exists(pf) or not info.get("n_admitted"):
                continue
            proj = np.load(pf).astype(np.float32)
            calib = np.load(os.path.join(vdir, f"{sname}.calib.npy")).astype(np.float32)
            ov = os.path.join(vdir, f"{sname}.overlay.npy")
            ext.append((f"robot.semvec.{sname}.proj", proj))
            ext.append((f"robot.semvec.{sname}.calib", calib))
            if os.path.exists(ov):
                ext.append((f"robot.semvec.{sname}.overlay", np.load(ov).astype(np.float32)))
            sv_sites.append((sname, info))

        if sv_sites:
            if "semvec" not in features:
                writer.add_array("therobot.features_optional", ["semvec"])
            writer.add_string("therobot.semvec.version", str(cvec["semvec"]["version"]))
            writer.add_string("therobot.semvec.hash", str(cvec["semvec"]["hash"]))
            writer.add_uint32("therobot.semvec.named_dim", int(named_dim))
            writer.add_uint32("therobot.semvec.latent_dim", int(latent_dim))
            if axis_names:
                writer.add_array("therobot.semvec.axes", axis_names)
            writer.add_uint32("therobot.semvec.site_count", len(sv_sites))
            site_meta = {s["name"]: s for s in cfg.candidate_sites}
            for i, (sname, info) in enumerate(sv_sites):
                p = f"therobot.semvec.site.{i}."
                sm = site_meta.get(sname, {})
                writer.add_string(p + "name", sname)
                if sm:
                    writer.add_uint32(p + "layer", int(sm["layer"]))
                    writer.add_string(p + "point", sm["point"])
                    writer.add_uint32(p + "offset", int(sm["offset"]))
                    writer.add_uint32(p + "width", int(sm["width"]))
                writer.add_uint32(p + "n_admitted", int(info["n_admitted"]))
                writer.add_uint32(p + "n_writable", int(info.get("n_writable", 0)))

    # §1.3/§1.4 state + modulator grafts (R3; zero-init when the graft stage
    # hasn't trained them — function-preserving by construction)
    graft = lock.section("graft")
    graft_dir = cfg.resolve(graft["tensor_dir"]) if graft else None

    def graft_tensor(name: str, shape: tuple) -> np.ndarray:
        if graft_dir:
            f = os.path.join(graft_dir, name.replace("/", "_") + ".npy")
            if os.path.exists(f):
                return np.load(f).astype(np.float32)
        return np.zeros(shape, dtype=np.float32)

    n_embd = int(reader.fields[f"{cfg.base_architecture}.embedding_length"].contents())

    if "state" in features:
        banks = cfg.state_banks
        s_width = sum(int(b["width"]) for b in banks)
        writer.add_uint32("therobot.state.bank_count", len(banks))
        for bi, bank in enumerate(banks):
            writer.add_string(f"therobot.state.bank.{bi}.name", bank["name"])
            writer.add_uint32(f"therobot.state.bank.{bi}.width", int(bank["width"]))
        writer.add_array("therobot.state.layers", [int(x) for x in cfg.state_layers])
        for layer in cfg.state_layers:
            ext.append((f"blk.{layer}.robot_state.alpha",
                        graft_tensor(f"blk.{layer}.robot_state.alpha", (s_width,))))
            ext.append((f"blk.{layer}.robot_state.in_proj.weight",
                        graft_tensor(f"blk.{layer}.robot_state.in_proj.weight", (s_width, n_embd))))
            ext.append((f"blk.{layer}.robot_state.out_proj.weight",  # zero at graft
                        graft_tensor(f"blk.{layer}.robot_state.out_proj.weight", (n_embd, s_width))))

    if "modulator" in features:
        mod = cfg.modulator
        m = int(mod["dim"])
        writer.add_uint32("therobot.modulator.dim", m)
        writer.add_array("therobot.modulator.channels", list(mod["channels"])[:m])
        writer.add_string("therobot.modulator.source", mod.get("source", "pooled"))
        pool_in = sum(int(b["width"]) for b in cfg.state_banks
                      if b["name"] == "glacial") if mod.get("source") == "glacial" else n_embd
        ext.append(("robot.mod.alpha", graft_tensor("robot.mod.alpha", (m,))))
        ext.append(("robot.mod.pool.weight", graft_tensor("robot.mod.pool.weight", (m, pool_in))))
        ext.append(("robot.mod.cell.weight", graft_tensor("robot.mod.cell.weight", (m, m))))
        for layer in graft.get("film_layers", []):
            gname = f"blk.{layer}.robot_film.gamma"
            bname = f"blk.{layer}.robot_film.beta"
            ext.append((gname + ".weight", graft_tensor(gname + ".weight", (n_embd, m))))
            gb = graft_tensor(gname + ".bias", (n_embd,))
            if not gb.any():
                gb = np.ones(n_embd, dtype=np.float32)  # γ ≡ 1 at graft
            ext.append((gname + ".bias", gb))
            ext.append((bname + ".weight", graft_tensor(bname + ".weight", (n_embd, m))))

    # §1.5 memory heads
    if "memory" in features:
        memc = cfg.raw.get("memory", {})
        d_in = sum(int(bn["width"]) for bn in lock.require("cleave", "cleave")["bottlenecks"])
        kd, vd = int(memc.get("key_dim", 16)), int(cfg.modulator["dim"])
        writer.add_uint32("therobot.memory.key_dim", kd)
        writer.add_uint32("therobot.memory.value_dim", vd)
        writer.add_uint32("therobot.memory.capacity", int(memc.get("capacity", 64)))
        writer.add_float32("therobot.memory.decay_halflife", float(memc.get("decay_halflife", 256.0)))
        writer.add_float32("therobot.memory.salience.threshold_quantile",
                           float(lock.section("shims").get("salience", {}).get("threshold_quantile", 0.9)))
        # absolute salience floor: writes must clear BOTH the running quantile and
        # this fixed bar, so a quiet stretch writes nothing (rare + meaningful).
        sal = memc.get("salience", {}) if isinstance(memc.get("salience"), dict) else {}
        writer.add_float32("therobot.memory.salience.floor", float(sal.get("floor", 0.0)))
        ext.append(("robot.mem.summary.key.weight", graft_tensor("robot.mem.summary.key.weight", (kd, d_in))))
        ext.append(("robot.mem.summary.value.weight", graft_tensor("robot.mem.summary.value.weight", (vd, d_in))))
        # Salience weight vector. Surprise is only the bootstrap signal; importance
        # also comes from the modulator's emotional channels. Emit [w_surprise,
        # w_mnorm] and, when the config names per-channel importance weights, the
        # extended [2+M] form so a calm-but-consequential moment (high threat /
        # valence, low surprise) is retained. Omitted → runtime defaults to 1,1.
        w_surprise = float(sal.get("surprise", 1.0))
        w_mnorm    = float(sal.get("mnorm", 1.0))
        chan_w = sal.get("channels", {}) or {}
        if chan_w:
            names = list(cfg.modulator.get("channels", []))
            vec = [w_surprise, w_mnorm] + [float(chan_w.get(n, 0.0)) for n in names]
            if len(vec) != 2 + vd:
                raise ValueError(
                    f"salience.channels needs one weight per modulator channel "
                    f"({vd} channels: {names}); got vector length {len(vec)}")
        else:
            vec = [w_surprise, w_mnorm]
        ext.append(("robot.mem.salience.weight", np.array(vec, dtype=np.float32)))

    # §1.6 delta thresholds (R4 — measured)
    if "delta" in features:
        cal = lock.require("calibrate", "calibrate")
        writer.add_string("therobot.delta.granularity", cal["granularity"])
        writer.add_uint32("therobot.delta.heartbeat", int(cal["heartbeat"]))
        writer.add_float32("therobot.delta.target_keep_rate", float(cal["target_keep_rate"]))
        for blk in cal["blocks"]:
            ext.append((f"blk.{blk['layer']}.robot_delta.theta_base",
                        np.array([blk["theta_base"]], dtype=np.float32)))

    # §1.7 settle
    if "settle" in features:
        st = cfg.settle
        writer.add_string("therobot.settle.objective", st.get("objective", "jacobi-ar"))
        writer.add_uint32("therobot.settle.mask_token_id", int(st.get("mask_token_id", 0)))
        writer.add_uint32("therobot.settle.max_steps", int(st.get("max_steps", 64)))
        writer.add_float32("therobot.settle.epsilon", float(st.get("epsilon", 0.0)))
        writer.add_array("therobot.settle.m_schedule",
                         [float(x) for x in st.get("m_schedule", [0.0])])

    ext = [(name, np.ascontiguousarray(data, dtype=np.float32)) for name, data in ext]
    for name, data in ext:
        writer.add_tensor_info(name, data.shape, data.dtype, data.nbytes,
                               gguf.GGMLQuantizationType.F32)

    _write_tensor_data(reader, writer, ext)
    lock.update("export", {"output": out_path, "features": features,
                           "extension_tensors": [n for n, _ in ext]})
    print(f"export: {out_path} — features {features}, {len(ext)} extension tensor(s)")


def strip(cfg: Config, in_path: str, out_path: str) -> None:
    """Downgrade an extended file to a stock base-arch file (spec §3)."""
    gguf = cfg.gguf_module()
    reader = gguf.GGUFReader(in_path)

    base_arch = reader.fields["therobot.base_architecture"].contents()
    writer = gguf.GGUFWriter(out_path, base_arch)

    for field in reader.fields.values():
        if (field.name == gguf.Keys.General.ARCHITECTURE or
                field.name.startswith("GGUF.") or field.name.startswith("therobot.")):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)

    def is_ext(name: str) -> bool:
        return name.startswith("robot.") or ".robot_" in name

    kept = [t for t in reader.tensors if not is_ext(t.name)]
    for tensor in kept:
        writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype,
                               tensor.data.nbytes, tensor.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for tensor in kept:
        writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)
    writer.close()
    print(f"strip: {out_path} — stock '{base_arch}' file, "
          f"{len(reader.tensors) - len(kept)} extension tensor(s) dropped")
