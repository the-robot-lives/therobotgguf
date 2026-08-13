"""shim-compile — compile semvec-DEFINED shims for this donor
(extraction-v1 §4.5: define once in the standard, compile per donor).

A semvec shim's source definition is donor-independent:

    semvec_shims:
      - { name: more-formal,  axis: formality, scale: +1.0, tags: [style] }
      - { name: damp-menace,  axis: menace,    scale: -1.5, tags: [safety],
          site: resid18,      max_crosstalk_ratio: 0.25 }

Compilation for donor@site is the axis's row of that site's write-calibrated
overlay G (overlay.py): steer = scale · G[axis]. Because G is calibrated
against the site's own encoder, the compiled edit moves the site's readout
of the axis by exactly `scale` — the admission question reduces to WRITE
CROSSTALK (how much the edit moves every *other* admitted axis), which
cleave-vec already measured and the exact affine algebra makes precise:
off-target shift = |scale| · roundtrip[axis, other]. No recordings and no
donor forward pass are needed to compile or to admit — that is the point.

Output: one `therobot-shim` GGUF per admitted definition (same module format
R5 emits, loadable by the E3/E8 machinery) with `therobot.shim.semvec.*`
provenance KVs, plus registry.json entries keyed by semvec version+hash.
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import SPEC_VERSION
from .config import Config, Lockfile
from .semvec import SemvecSpec


def _pick_site(cvec: dict, axis: str, want: str = "") -> tuple:
    """Best admitted+writable site for the axis: honor an explicit `site:`,
    else highest decodability with lowest crosstalk as the tiebreak."""
    cands = []
    for sname, info in cvec["sites"].items():
        for a in info["axes"]:
            if a["axis"] == axis and a.get("writable"):
                cands.append((sname, a))
    if want:
        cands = [(s, a) for s, a in cands if s == want]
    if not cands:
        return None, None
    cands.sort(key=lambda sa: (-sa[1]["decodability"], sa[1].get("write_crosstalk", 1.0)))
    return cands[0]


def run(cfg: Config) -> None:
    lock = Lockfile(cfg.lockfile_path)
    cvec = lock.require("cleave_vec", "cleave")
    if not cfg.semvec:
        raise SystemExit("shim-compile: config names no `semvec:` spec")
    spec = SemvecSpec.load(cfg.resolve(cfg.semvec))
    if cvec["semvec"]["hash"] != spec.spec_hash():
        raise SystemExit(f"shim-compile: lockfile semvec hash {cvec['semvec']['hash']} "
                         f"!= spec {spec.spec_hash()} — coordinate systems must match")

    defs = cfg.raw.get("semvec_shims") or []
    if not defs:
        raise SystemExit("shim-compile: config has no `semvec_shims:` definitions")
    gguf = cfg.gguf_module()
    vdir = cfg.resolve(cvec["probe_dir"])
    out_dir = os.path.join(cfg.workdir, "modules")
    os.makedirs(out_dir, exist_ok=True)

    names = [a.name for a in spec.axes] + [f"latent_{i}" for i in range(spec.latent_dim)]
    idx_of = {n: i for i, n in enumerate(names)}

    admitted, rejected = [], []
    for d in defs:
        name, axis = d["name"], d["axis"]
        scale = float(d.get("scale", 1.0))
        if axis not in idx_of:
            rejected.append({"name": name, "reason": f"unknown semvec axis '{axis}'"})
            continue
        sname, ainfo = _pick_site(cvec, axis, want=d.get("site", ""))
        if sname is None:
            rejected.append({"name": name,
                             "reason": f"no admitted+writable site for '{axis}'"
                                       + (f" (site {d['site']})" if d.get("site") else "")})
            continue

        proj = np.load(os.path.join(vdir, f"{sname}.proj.npy")).astype(np.float32)
        g = np.load(os.path.join(vdir, f"{sname}.overlay.npy")).astype(np.float32)
        j = idx_of[axis]
        steer = (scale * g[j]).astype(np.float32)          # [site width]

        # admission by exact affine algebra: readout shifts under the edit
        shift = steer @ proj                                # [D]
        effect = float(shift[j])                            # ≈ scale (calibrated)
        others = np.abs(np.delete(shift, j))
        off_target = float(others.max()) if len(others) else 0.0
        ratio = off_target / max(abs(effect), 1e-9)
        max_ratio = float(d.get("max_crosstalk_ratio", 0.25))
        if abs(effect - scale) > 0.05 * abs(scale) or ratio > max_ratio:
            rejected.append({"name": name, "site": sname,
                             "effect": round(effect, 4),
                             "off_target": round(off_target, 4),
                             "reason": f"crosstalk ratio {ratio:.3f} > {max_ratio}"
                             if ratio > max_ratio else "calibration drift"})
            continue

        path = os.path.join(out_dir, f"{name}.gguf")
        w = gguf.GGUFWriter(path, "therobot-shim")
        w.add_uint32("therobot.spec_version", SPEC_VERSION)
        w.add_string("therobot.shim.name", name)
        w.add_string("therobot.shim.version", "0.1.0")
        w.add_string("therobot.shim.target_model", cfg.config_hash())
        w.add_string("therobot.shim.target_bottleneck", sname)
        w.add_string("therobot.shim.effect",
                     d.get("effect", f"{'+' if scale >= 0 else ''}{scale:g} {axis} (semvec)"))
        w.add_float32("therobot.shim.selectivity", float(round(1.0 - ratio, 6)))
        w.add_string("therobot.shim.gate", d.get("gate", "always"))
        w.add_array("therobot.shim.depends", list(d.get("depends", [])))
        w.add_array("therobot.shim.conflicts", list(d.get("conflicts", [])))
        # semvec provenance — the portable identity of this module
        w.add_string("therobot.shim.semvec.version", str(spec.version))
        w.add_string("therobot.shim.semvec.hash", spec.spec_hash())
        w.add_string("therobot.shim.semvec.axis", axis)
        w.add_float32("therobot.shim.semvec.scale", scale)
        w.add_tensor("robot.shim.steer", steer)
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()

        admitted.append({"name": name, "file": os.path.basename(path),
                         "tags": list(d.get("tags", [axis])),
                         "selectivity": float(round(1.0 - ratio, 6)),
                         "semvec": {"version": spec.version, "hash": spec.spec_hash(),
                                    "axis": axis, "scale": scale, "site": sname},
                         "depends": list(d.get("depends", [])),
                         "conflicts": list(d.get("conflicts", []))})
        print(f"shim-compile: admitted '{name}' → {sname} "
              f"(effect {effect:+.3f} on {axis}, worst off-target {off_target:.3f})")

    reg_path = os.path.join(out_dir, "registry.json")
    reg = {"spec_version": SPEC_VERSION, "model": cfg.config_hash(), "shims": []}
    if os.path.exists(reg_path):
        with open(reg_path) as f:
            reg = json.load(f)
    have = {s["name"] for s in admitted}
    reg["shims"] = [s for s in reg.get("shims", []) if s["name"] not in have] + admitted
    reg["semvec"] = {"version": spec.version, "hash": spec.spec_hash()}
    with open(reg_path, "w") as f:
        json.dump(reg, f, indent=2)

    lock.update("shim_compile", {"admitted": admitted, "rejected": rejected,
                                 "registry": reg_path})
    print(f"shim-compile: {len(admitted)} admitted, {len(rejected)} rejected "
          f"→ {reg_path}")
