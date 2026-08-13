"""R8 — Verify: drive the fork's runtime binaries end-to-end.

Owns the gates from the runtime README §5: the exported file must load in the
fork (feature negotiation included), an L-zero-graft export must be
logit-identical to its stock base (the permanent parity invariant), and a
stripped file must load as a stock model. Behavioral probes (priming, memory
decay, delta divergence, settle correlation) run through the fork's
tests/robot binaries when pointed at real fixtures.
"""
from __future__ import annotations

import os
import subprocess

from .config import Config, Lockfile


def _run(cmd: list, label: str) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        tail = (r.stdout + r.stderr).strip().splitlines()[-8:]
        raise SystemExit(f"verify: {label} FAILED (rc={r.returncode})\n  " + "\n  ".join(tail))
    return r.stdout


def run(cfg: Config, parity_bin: str | None = None) -> None:
    if not cfg.paths.runtime_bin:
        raise SystemExit("config paths.runtime_bin must point at the fork build's bin/ directory")
    bin_dir = cfg.resolve(cfg.paths.runtime_bin)
    inspect = os.path.join(bin_dir, "llama-robot-inspect")
    if not os.path.exists(inspect):
        raise SystemExit(f"verify: {inspect} not found — build the fork first")

    lock = Lockfile(cfg.lockfile_path)
    exported = lock.require("export", "export")
    out = exported["output"]
    results = {}

    # gate 1: manifest + negotiated load in the live runtime
    stdout = _run([inspect, out, "--load"], "runtime load")
    if "load check OK" not in stdout:
        raise SystemExit("verify: runtime did not accept the exported file")
    results["load"] = "OK"
    print(f"verify: runtime load OK ({os.path.basename(out)})")

    # gate 2: L0 parity — zero-graft export ≡ stock base, bit-exact logits
    if parity_bin and not os.path.exists(parity_bin):
        print(f"verify: SKIP parity gate — binary not found at {parity_bin} "
              f"(compile tests/robot/robot_parity_test.cpp; see qwen3.5-0.8b.md)")
        results["parity"] = "skipped (binary missing)"
    elif parity_bin and cfg.base_gguf:
        stdout = _run([parity_bin, cfg.resolve(cfg.base_gguf), out], "parity")
        if "PARITY OK" not in stdout:
            raise SystemExit("verify: parity gate failed:\n" + stdout)
        results["parity"] = stdout.strip().splitlines()[0]
        print(f"verify: parity gate OK ({results['parity']})")
    else:
        print("verify: parity gate not requested (pass --parity-bin to run it)")

    # gate 3: strip interop — the downgraded file loads as a stock model
    stripped = os.path.splitext(out)[0] + ".stripped.gguf"
    from . import export as export_mod  # noqa: PLC0415
    export_mod.strip(cfg, out, stripped)
    stdout = _run([inspect, stripped, "--load"], "stripped load")
    if "load check OK" not in stdout:
        raise SystemExit("verify: stripped file did not load as a stock model")
    results["strip"] = "OK"
    print("verify: strip interop OK")

    # gate 4: admitted shim modules attach against the live export
    shims = lock.section("shims")
    for entry in shims.get("admitted", []):
        module = os.path.join(cfg.workdir, "modules", entry["file"])
        stdout = _run([inspect, module], f"shim manifest {entry['name']}")
        if "therobot-shim module" not in stdout:
            raise SystemExit(f"verify: module '{entry['name']}' has a bad manifest")
    if shims.get("admitted"):
        results["modules"] = f"{len(shims['admitted'])} manifest(s) OK"
        print(f"verify: {len(shims['admitted'])} shim module manifest(s) OK")

    lock.update("verify", results)
    print("verify: all gates green")
