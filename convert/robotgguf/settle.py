"""R6 — Settling track (004). Separate, optional, expensive.

v0 policy (conversion-pipeline.md): don't convert the AR donor — run an
existing diffusion checkpoint (Dream/LLaDA-class) through R0–R5 instead, and
let the runtime's settle decoder drive it. The runtime currently implements
the `jacobi-ar` objective (AR fixed-point settling), which works on any
causal donor; `mdlm` waits on the diffusion donor path. This stage only
validates and records the settle configuration.
"""
from __future__ import annotations

from .config import Config, Lockfile


def run(cfg: Config) -> None:
    lock = Lockfile(cfg.lockfile_path)
    st = dict(cfg.settle)
    if not st.get("enabled"):
        print("settle: disabled in config — nothing to do")
        return
    objective = st.get("objective", "jacobi-ar")
    if objective not in ("jacobi-ar", "mdlm"):
        raise SystemExit(f"settle: unknown objective '{objective}'")
    if objective == "mdlm":
        print("settle: NOTE — 'mdlm' exports will be refused by the current runtime; "
              "route a Dream/LLaDA-class donor through R0–R5 when that path lands")
    lock.update("settle", {
        "objective": objective,
        "mask_token_id": int(st.get("mask_token_id", 0)),
        "max_steps": int(st.get("max_steps", 64)),
        "epsilon": float(st.get("epsilon", 0.0)),
        "m_schedule": [float(x) for x in st.get("m_schedule", [0.0, 1.0, 2.0, 4.0])],
    })
    print(f"settle: configured objective '{objective}'")
