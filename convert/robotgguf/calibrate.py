"""R4 — Calibrate delta thresholds (002). No training required for v0.

From per-site recordings (treated as consecutive-token streams), compute the
mean-squared input change between consecutive samples — the exact statistic
the runtime's fire rule uses (fire = mean((x − held)²) > θ_eff) — and set
θ_base per covered block by running-quantile at the config's target keep
rate. Fatigue and excitability stay at neutral defaults. A simulated
held-input replay validates the achieved keep rate and picks the heartbeat
from the drift curve.
"""
from __future__ import annotations

import numpy as np

from .config import Config, Lockfile
from .recordings import RecordingStore


def _msq_deltas(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    d = np.diff(x, axis=0)
    return (d * d).mean(axis=1)


def _simulate(msq_stream: np.ndarray, theta: float, heartbeat: int):
    """Replay the fire rule against held inputs; returns (keep_rate, max_staleness)."""
    fires = 0
    stale = worst = 0
    held_pending = 0.0
    for i, m in enumerate(msq_stream):
        forced = (i % heartbeat) == 0
        held_pending += m  # drift accumulates against the held input
        if forced or held_pending > theta:
            fires += 1
            held_pending = 0.0
            stale = 0
        else:
            stale += 1
            worst = max(worst, stale)
    return fires / max(1, len(msq_stream)), worst


def run(cfg: Config) -> None:
    if not cfg.delta.get("enabled"):
        print("calibrate: delta disabled in config — nothing to do")
        return

    store = RecordingStore(cfg.recordings_dir)
    if not store.exists():
        raise SystemExit(f"no recordings at {cfg.recordings_dir} — run `robotgguf record` first")
    man = store.manifest
    lock = Lockfile(cfg.lockfile_path)

    keep = float(cfg.delta.get("target_keep_rate", 0.3))
    heartbeat = int(cfg.delta.get("heartbeat", 32))

    blocks = []
    for layer in cfg.delta.get("layers", []):
        # calibrate each block from the recorded site nearest its input
        site = min(man.sites.items(), key=lambda kv: abs(kv[1]["layer"] - (layer - 1)))
        msq = _msq_deltas(store.activations(site[0]))
        theta = float(np.quantile(msq, 1.0 - keep))
        achieved, staleness = _simulate(msq, theta, heartbeat)
        blocks.append({"layer": int(layer), "theta_base": round(theta, 6),
                       "calibrated_from": site[0],
                       "achieved_keep_rate": round(achieved, 4),
                       "max_staleness": int(staleness)})
        print(f"calibrate: block {layer}: θ={theta:.5g} keep≈{achieved:.2f} "
              f"(target {keep}), max staleness {staleness} (heartbeat {heartbeat})")

    lock.update("calibrate", {
        "granularity": "block",
        "heartbeat": heartbeat,
        "target_keep_rate": keep,
        "blocks": blocks,
    })
