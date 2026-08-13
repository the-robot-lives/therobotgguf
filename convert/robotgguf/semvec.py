"""semvec — the standardized semantic label vector (extraction-v1 §3.2, §4.4).

Loads and validates a semvec spec (configs/semvec-v1.yaml), assembles the
per-position label vector from tier outputs, derives the v0 categorical views,
and provides the query-side embedding reduction used by both labeling (Block B
targets) and the runtime's zero-shot mode (the SAME frozen map, by
construction — that identity is what makes `semvec_query` meaningful).

The spec is a versioned, donor-independent STANDARD: axes are append-only,
and `spec_hash` pins everything that defines the coordinate system.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field

import numpy as np
import yaml


@dataclass
class Axis:
    index: int
    name: str
    group: str
    tier: str          # preferred owner: t0 | t1 | t2


@dataclass
class SemvecSpec:
    path: str
    version: str
    named_dim: int
    latent_dim: int
    scale: tuple
    axes: list                      # [Axis], sorted by index, len == named_dim
    views: dict                     # attr → view spec dict
    latent: dict                    # embedder identity + frozen basis refs
    by_name: dict = field(default_factory=dict)

    @property
    def dim(self) -> int:
        return self.named_dim + self.latent_dim

    def axis(self, name: str) -> Axis:
        try:
            return self.by_name[name]
        except KeyError:
            raise SystemExit(f"semvec: unknown axis '{name}' (spec {self.version})")

    def spec_hash(self) -> str:
        """Identity of the coordinate system: version, axis order/names,
        scale, dims, and the frozen latent identity."""
        h = hashlib.sha256()
        h.update(self.version.encode())
        h.update(f"{self.named_dim}:{self.latent_dim}:{self.scale}".encode())
        for a in self.axes:
            h.update(f"{a.index}:{a.name}:{a.group}".encode())
        lat = self.latent or {}
        h.update(str(lat.get("embedder")).encode())
        h.update(str(lat.get("basis_sha256")).encode())
        return h.hexdigest()[:16]

    @staticmethod
    def load(path: str) -> "SemvecSpec":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        core = raw.get("semvec") or {}
        named_dim = int(core.get("named_dim", 128))
        latent_dim = int(core.get("latent_dim", 384))
        tiers = raw.get("tiers") or {}

        axes: list = []
        for group, g in (raw.get("groups") or {}).items():
            start = int(g["start"])
            tier = tiers.get(group, "t2")
            for i, name in enumerate(g["axes"]):
                axes.append(Axis(index=start + i, name=name, group=group, tier=tier))
        axes.sort(key=lambda a: a.index)

        # validate: contiguous, unique, exactly named_dim
        idxs = [a.index for a in axes]
        if idxs != list(range(named_dim)):
            missing = sorted(set(range(named_dim)) - set(idxs))
            dupes = sorted({i for i in idxs if idxs.count(i) > 1})
            raise SystemExit(f"semvec: axis indices must cover 0..{named_dim - 1} "
                             f"exactly (missing {missing[:8]}, duplicated {dupes[:8]})")
        names = [a.name for a in axes]
        if len(set(names)) != len(names):
            raise SystemExit("semvec: duplicate axis names")

        spec = SemvecSpec(path=os.path.abspath(path),
                          version=str(core.get("version", "1.0")),
                          named_dim=named_dim, latent_dim=latent_dim,
                          scale=tuple(core.get("scale", [0.0, 4.0])),
                          axes=axes, views=raw.get("views") or {},
                          latent=raw.get("latent") or {})
        spec.by_name = {a.name: a for a in spec.axes}

        # validate views reference real axes
        for attr, v in spec.views.items():
            for name in ([v.get("axis")] if "axis" in v else v.get("axes", [])):
                if name and name not in spec.by_name:
                    raise SystemExit(f"semvec: view '{attr}' references unknown axis '{name}'")
        return spec


# ---- vector assembly ----

class VectorBuilder:
    """Accumulates per-axis scores from tiers into the [N, D] vector.

    Tier precedence is t0 < t1 < t2: a later (higher) tier overwrites, a
    lower tier never clobbers a higher one. Who wrote each axis is recorded
    for the lockfile / vector_sources.json.
    """

    _RANK = {"t0": 0, "t1": 1, "t2": 2}

    def __init__(self, spec: SemvecSpec, n: int):
        self.spec = spec
        self.vec = np.zeros((n, spec.dim), dtype=np.float32)
        self.sources: dict = {}          # axis name → tier that wrote it

    def set_axis(self, name: str, values: np.ndarray, tier: str) -> None:
        ax = self.spec.axis(name)
        cur = self.sources.get(name)
        if cur is not None and self._RANK[cur] >= self._RANK[tier]:
            return
        lo, hi = self.spec.scale
        self.vec[:, ax.index] = np.clip(np.asarray(values, dtype=np.float32), lo, hi)
        self.sources[name] = tier

    def set_latent(self, block: np.ndarray, tier: str = "t1") -> None:
        if block.shape[1] != self.spec.latent_dim:
            raise SystemExit(f"semvec: latent block width {block.shape[1]} != "
                             f"spec latent_dim {self.spec.latent_dim}")
        self.vec[:, self.spec.named_dim:] = block
        self.sources["__latent__"] = tier

    def finish(self) -> np.ndarray:
        return self.vec.astype(np.float16)


# ---- categorical views (the v0 seven, reconstructed) ----

def categorical_views(spec: SemvecSpec, vec: np.ndarray, attrs: list) -> dict:
    """Derive int64 categorical labels from the named block. `vec` may be
    [N, D] float16/32; only named axes are read."""
    v = np.asarray(vec, dtype=np.float32)
    out: dict = {}
    for attr in attrs:
        view = spec.views.get(attr)
        if view is None:
            raise SystemExit(f"semvec: no categorical view for attribute '{attr}'")
        kind = view.get("kind")
        if kind == "threshold":
            x = v[:, spec.axis(view["axis"]).index]
            y = np.zeros(len(v), dtype=np.int64)
            for t in view["thresholds"]:
                y += (x >= float(t)).astype(np.int64)
        elif kind == "argmax":
            cols = np.stack([v[:, spec.axis(a).index] for a in view["axes"]], axis=1)
            y = np.argmax(cols, axis=1).astype(np.int64)
            below = cols.max(axis=1) < float(view.get("floor", -np.inf))
            fb = view.get("fallback")
            y[below] = int(fb) if fb is not None else len(view["axes"])
        else:
            raise SystemExit(f"semvec: view '{attr}' has unknown kind '{kind}'")
        out[attr] = y
    return out


# ---- the frozen latent reduction (labels AND queries go through this) ----

def load_basis(spec: SemvecSpec, resolve=lambda p: p):
    """Load the frozen PCA/whitening basis (mean [E], components [E, latent_dim]).
    Returns None when the latent block isn't frozen yet."""
    ref = (spec.latent or {}).get("basis")
    if not ref:
        return None
    path = resolve(ref)
    with np.load(path) as z:
        mean, comp = z["mean"], z["components"]
    want = (spec.latent or {}).get("basis_sha256")
    if want:
        got = hashlib.sha256(open(path, "rb").read()).hexdigest()
        if got != want:
            raise SystemExit(f"semvec: basis {path} sha256 mismatch — the latent "
                             f"coordinate system is pinned; refusing (got {got[:12]}…)")
    if comp.shape[1] != spec.latent_dim:
        raise SystemExit(f"semvec: basis produces {comp.shape[1]} dims, "
                         f"spec latent_dim is {spec.latent_dim}")
    return mean.astype(np.float32), comp.astype(np.float32)


def reduce_embeddings(emb: np.ndarray, basis) -> np.ndarray:
    """Embedder output → Block-B coordinates. Used identically for corpus
    sentences (labels) and for query texts (zero-shot readout)."""
    mean, comp = basis
    return (np.asarray(emb, dtype=np.float32) - mean) @ comp


def fit_basis(emb: np.ndarray, latent_dim: int, out_path: str) -> str:
    """Fit and freeze the reduction: PCA (via SVD) with whitening. Returns the
    sha256 to pin in the spec. Fitting twice on different samples is a NEW
    coordinate system — extraction-v1's append-only rule applies."""
    x = np.asarray(emb, dtype=np.float32)
    if len(x) < latent_dim:
        raise SystemExit(f"semvec: need ≥{latent_dim} embeddings to fit the basis, got {len(x)}")
    mean = x.mean(axis=0)
    xc = x - mean
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    comp = (vt[:latent_dim] / (s[:latent_dim, None] / np.sqrt(len(x) - 1) + 1e-8)).T
    np.savez(out_path, mean=mean.astype(np.float32), components=comp.astype(np.float32))
    return hashlib.sha256(open(out_path, "rb").read()).hexdigest()


def save_sources(recordings_dir: str, spec: SemvecSpec, sources: dict) -> None:
    with open(os.path.join(recordings_dir, "labels", "vector_sources.json"), "w") as f:
        json.dump({"semvec_version": spec.version, "semvec_hash": spec.spec_hash(),
                   "sources": sources}, f, indent=2, sort_keys=True)
