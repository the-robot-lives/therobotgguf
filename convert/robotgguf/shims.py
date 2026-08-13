"""R5 — Shims (§E): slice-scoped steering modules trained against recordings
only — the donor core is never loaded (005 §2.3).

v0 shim form: a steering vector on the bottleneck slice, the difference of
class-conditional means scaled by the config's strength. Admission (005
§2.4): the R2 probe must report the edit moves the target attribute
(effect margin) while sibling attributes hold (selectivity); admitted shims
export as standalone `therobot-shim` GGUFs plus a registry.json entry that
the runtime's E8 router consumes directly.
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import SPEC_VERSION
from .config import Config, Lockfile
from .recordings import RecordingStore


def _probe_shift(probe_w: np.ndarray, probe_b: np.ndarray, x: np.ndarray,
                 steer: np.ndarray, target_class: int) -> float:
    """Mean increase in the probe's target-class probability under the edit."""
    def prob(v):
        z = v @ probe_w.T + probe_b
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return (e / e.sum(axis=1, keepdims=True))[:, target_class]
    return float(prob(x + steer).mean() - prob(x).mean())


def run(cfg: Config) -> None:
    lock = Lockfile(cfg.lockfile_path)
    cleaved = lock.require("cleave", "cleave")
    store = RecordingStore(cfg.recordings_dir)
    man = store.manifest
    probe_dir = cfg.resolve(cleaved["probe_dir"])
    gguf = cfg.gguf_module()

    out_dir = os.path.join(cfg.workdir, "modules")
    os.makedirs(out_dir, exist_ok=True)

    by_name = {bn["name"]: bn for bn in cleaved["bottlenecks"]}
    admitted, rejected = [], []

    for spec in cfg.shims:
        name = spec["name"]
        attr = spec["attribute"]
        target = int(spec.get("direction", 1))
        scale = float(spec.get("scale", 1.0))

        bn = next((b for b in cleaved["bottlenecks"] if attr in b["attributes"]), None)
        if bn is None:
            rejected.append({"name": name, "reason": f"no admitted bottleneck decodes '{attr}'"})
            continue

        x = np.asarray(store.activations(bn["name"]), dtype=np.float32)
        y = store.labels(attr)

        # steering vector: class-mean difference on the slice, recordings only
        steer = scale * (x[y == target].mean(0) - x[y != target].mean(0)).astype(np.float32)

        # admission: move the target attribute, hold the others (R2's probes)
        pw = np.load(os.path.join(probe_dir, f"{bn['name']}.{attr}.weight.npy"))
        pb = np.load(os.path.join(probe_dir, f"{bn['name']}.{attr}.bias.npy"))
        effect = _probe_shift(pw, pb, x, steer, target)

        hold_worst = 0.0
        for other in bn["attributes"]:
            if other == attr:
                continue
            ow = np.load(os.path.join(probe_dir, f"{bn['name']}.{other}.weight.npy"))
            ob = np.load(os.path.join(probe_dir, f"{bn['name']}.{other}.bias.npy"))
            oy = store.labels(other)
            for c in range(int(oy.max()) + 1):
                hold_worst = max(hold_worst, abs(_probe_shift(ow, ob, x, steer, c)))

        selectivity = effect - hold_worst
        if effect <= 0.0 or selectivity < float(spec.get("min_selectivity", 0.05)):
            rejected.append({"name": name, "effect": round(effect, 4),
                             "off_target": round(hold_worst, 4),
                             "reason": "failed admission (move target, hold others)"})
            continue

        # standalone module file (spec §4)
        path = os.path.join(out_dir, f"{name}.gguf")
        w = gguf.GGUFWriter(path, "therobot-shim")
        w.add_uint32("therobot.spec_version", SPEC_VERSION)
        w.add_string("therobot.shim.name", name)
        w.add_string("therobot.shim.version", "0.1.0")
        w.add_string("therobot.shim.target_model", man.model)
        w.add_string("therobot.shim.target_bottleneck", bn["name"])
        w.add_string("therobot.shim.effect", spec.get(
            "effect", f"steer '{attr}' toward class {target}"))
        w.add_float32("therobot.shim.selectivity", float(round(selectivity, 6)))
        w.add_string("therobot.shim.gate", spec.get("gate", "always"))
        w.add_array("therobot.shim.depends", list(spec.get("depends", [])))
        w.add_array("therobot.shim.conflicts", list(spec.get("conflicts", [])))
        w.add_tensor("robot.shim.steer", steer)
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()

        admitted.append({"name": name, "file": os.path.basename(path),
                         "tags": list(spec.get("tags", [attr])),
                         "selectivity": float(round(selectivity, 6)),
                         "depends": list(spec.get("depends", [])),
                         "conflicts": list(spec.get("conflicts", []))})
        print(f"shims: admitted '{name}' → {bn['name']} "
              f"(effect {effect:+.3f}, worst off-target {hold_worst:.3f})")

    with open(os.path.join(out_dir, "registry.json"), "w") as f:
        json.dump({"spec_version": SPEC_VERSION, "model": man.model,
                   "shims": admitted}, f, indent=2)

    # salience gate calibration (§F): quantile-normalized threshold provenance
    salience = {"threshold_quantile": float(cfg.raw.get("memory", {}).get(
        "salience_threshold_quantile", 0.9))}

    lock.update("shims", {"admitted": admitted, "rejected": rejected,
                          "registry": os.path.join(out_dir, "registry.json"),
                          "salience": salience})
    print(f"shims: {len(admitted)} admitted, {len(rejected)} rejected "
          f"→ {out_dir}/registry.json")
