"""Conversion config + lockfile.

The per-donor YAML is the single declarative artifact; stages append their
*measured* outputs to `<config>.lock.yaml` so downstream stages and the
exporter consume measured values, never hand-entered ones
(conversion-pipeline.md §1).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

import yaml


@dataclass
class Paths:
    gguf_py: Optional[str] = None       # fork's gguf-py (for R7); falls back to installed `gguf`
    runtime_bin: Optional[str] = None   # fork build dir with llama-robot-inspect etc. (for R8)
    recordings: str = "recordings"      # recording store root (relative to workdir)
    workdir: str = "work"               # stage outputs (lockfile lives next to the config)


@dataclass
class Config:
    path: str
    raw: dict

    # identity
    donor: str = ""                     # HF id or local path
    base_architecture: str = "llama"    # donor family (stock GGUF arch string)
    base_gguf: Optional[str] = None     # pre-converted stock GGUF (skips convert_hf_to_gguf)

    # R1/R2
    corpus: list = field(default_factory=list)     # text files / dataset specs
    candidate_sites: list = field(default_factory=list)  # [{name, layer, point, offset, width}]
    attributes: list = field(default_factory=list)       # attribute names (weak labels expected in recordings)

    # R2 selection targets
    max_bottlenecks: int = 8
    min_decodability: float = 0.7
    min_selectivity: float = 0.05

    # R2 v1 — semvec vector path (extraction-v1 §3-4). `semvec` names the
    # versioned spec (configs/semvec-v1.yaml); null keeps the pure-v0 path.
    semvec: Optional[str] = None
    min_axis_decodability: float = 0.3     # Spearman/Pearson on held-out
    min_axis_selectivity: float = 0.05     # vs row-shuffled control
    min_domain_stability_ratio: float = 0.9  # × min_axis_decodability
    mlp_fallback_max: int = 8              # nonlinear retries per site (findings)
    vec_l2: float = 10.0                   # ridge strength
    vec_sample_cap: int = 1_000_000        # solve/scoring sample bound

    # R3 graft
    state_banks: list = field(default_factory=lambda: [
        {"name": "fast", "width": 16}, {"name": "glacial", "width": 8}])
    state_layers: list = field(default_factory=list)
    modulator: dict = field(default_factory=lambda: {
        "dim": 8, "channels": ["arousal", "valence", "attention", "safety",
                               "novelty", "focus", "warmth", "energy"],
        "source": "pooled"})

    # R4 delta
    delta: dict = field(default_factory=lambda: {
        "enabled": False, "target_keep_rate": 0.3, "heartbeat": 32, "layers": []})

    # R5 shims
    shims: list = field(default_factory=list)  # [{name, attribute, direction, tags, scale}]

    # R6 settle
    settle: dict = field(default_factory=lambda: {"enabled": False})

    # R7
    features: list = field(default_factory=lambda: ["taps"])
    level: int = 1

    paths: Paths = field(default_factory=Paths)

    @staticmethod
    def load(path: str) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        cfg = Config(path=os.path.abspath(path), raw=raw)
        for key in ("donor", "base_architecture", "base_gguf", "corpus",
                    "candidate_sites", "attributes", "max_bottlenecks",
                    "min_decodability", "min_selectivity", "semvec",
                    "min_axis_decodability", "min_axis_selectivity",
                    "min_domain_stability_ratio", "mlp_fallback_max",
                    "vec_l2", "vec_sample_cap", "state_banks",
                    "state_layers", "modulator", "delta", "shims", "settle",
                    "features", "level"):
            if key in raw:
                setattr(cfg, key, raw[key])
        for key, val in (raw.get("paths") or {}).items():
            setattr(cfg.paths, key, val)
        return cfg

    # ---- resolved locations ----
    @property
    def root(self) -> str:
        return os.path.dirname(self.path)

    def resolve(self, p: str) -> str:
        # relative paths in the config (corpus/, work/, ../../../3rd-party/...)
        # are authored relative to the working directory the pipeline is run
        # from (the `convert/` package root), not the config file's directory
        return p if os.path.isabs(p) else os.path.abspath(p)

    @property
    def workdir(self) -> str:
        d = self.resolve(self.paths.workdir)
        os.makedirs(d, exist_ok=True)
        return d

    @property
    def recordings_dir(self) -> str:
        return self.resolve(self.paths.recordings)

    @property
    def lockfile_path(self) -> str:
        base, _ = os.path.splitext(self.path)
        return base + ".lock.yaml"

    def config_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.raw, sort_keys=True).encode()).hexdigest()[:16]

    def gguf_module(self):
        """Import gguf, preferring the fork's gguf-py (guarantees KV parity
        with the runtime that will load the export)."""
        if self.paths.gguf_py:
            p = self.resolve(self.paths.gguf_py)
            if p not in sys.path:
                sys.path.insert(0, p)
        import gguf  # noqa: PLC0415
        return gguf


class Lockfile:
    """Measured-values store, appended stage by stage."""

    def __init__(self, path: str):
        self.path = path
        self.data: dict[str, Any] = {}
        if os.path.exists(path):
            with open(path) as f:
                self.data = yaml.safe_load(f) or {}

    def section(self, name: str) -> dict:
        return self.data.get(name) or {}

    def update(self, name: str, values: dict) -> None:
        self.data[name] = values
        with open(self.path, "w") as f:
            yaml.safe_dump(self.data, f, sort_keys=False)

    def require(self, name: str, stage_hint: str) -> dict:
        sec = self.section(name)
        if not sec:
            raise SystemExit(f"lockfile has no '{name}' section — run `robotgguf {stage_hint}` first")
        return sec
