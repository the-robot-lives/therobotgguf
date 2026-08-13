"""Recording store (R1's output, everything else's training substrate).

Layout: <root>/<site>/act.npy (fp16 [n_samples, width]) per candidate site,
<root>/labels/<attribute>.npy (int64 [n_samples]), and manifest.json binding
{model hash, corpus hash, spec version, shard boundaries}. Recordings are
versioned contracts and regenerable derived data (005 §6).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

from . import SPEC_VERSION


@dataclass
class Manifest:
    model: str
    corpus: str
    spec_version: int
    n_samples: int
    sites: dict          # name → {layer, point, offset, width}
    attributes: list
    shards: list         # sample-index boundaries, for stability scoring
    # extraction-v1 additions (defaults keep pre-v1 manifests loadable):
    domain_names: list = None    # stratum names; labels/domain.npy indexes these
    semvec: dict = None          # {version, hash} when labels/vector.npy exists

    def save(self, root: str) -> None:
        with open(os.path.join(root, "manifest.json"), "w") as f:
            json.dump(self.__dict__, f, indent=2)

    @staticmethod
    def load(root: str) -> "Manifest":
        with open(os.path.join(root, "manifest.json")) as f:
            return Manifest(**json.load(f))


class RecordingStore:
    def __init__(self, root: str):
        self.root = root

    def exists(self) -> bool:
        return os.path.exists(os.path.join(self.root, "manifest.json"))

    @property
    def manifest(self) -> Manifest:
        return Manifest.load(self.root)

    # ---- write (R1 / synthetic fixtures) ----
    def write(self, model: str, corpus: str, sites: dict, acts: dict,
              labels: dict, n_shards: int = 4, domain_names: list = None,
              semvec: dict = None) -> None:
        os.makedirs(os.path.join(self.root, "labels"), exist_ok=True)
        n = None
        for name, a in acts.items():
            a = np.asarray(a, dtype=np.float16)
            n = len(a) if n is None else n
            assert len(a) == n, f"site {name}: sample count mismatch"
            os.makedirs(os.path.join(self.root, name), exist_ok=True)
            np.save(os.path.join(self.root, name, "act.npy"), a)
        for attr, y in labels.items():
            y = np.asarray(y, dtype=np.int64)
            assert len(y) == n, f"labels {attr}: sample count mismatch"
            np.save(os.path.join(self.root, "labels", f"{attr}.npy"), y)
        if semvec is None and os.path.exists(os.path.join(self.root, "labels", "vector_sources.json")):
            with open(os.path.join(self.root, "labels", "vector_sources.json")) as f:
                src = json.load(f)
            semvec = {"version": src.get("semvec_version"), "hash": src.get("semvec_hash")}
        bounds = [int(i * n / n_shards) for i in range(n_shards)] + [n]
        Manifest(model=model, corpus=corpus, spec_version=SPEC_VERSION,
                 n_samples=n, sites=sites, attributes=sorted(labels),
                 shards=bounds, domain_names=domain_names, semvec=semvec).save(self.root)

    # ---- read (R2/R4/R5) ----
    def activations(self, site: str) -> np.ndarray:
        return np.load(os.path.join(self.root, site, "act.npy"), mmap_mode="r")

    def labels(self, attribute: str) -> np.ndarray:
        return np.load(os.path.join(self.root, "labels", f"{attribute}.npy"))

    def label_vector(self) -> np.ndarray:
        """The semvec label vector [N, D] float16, or None (pre-v1 store)."""
        path = os.path.join(self.root, "labels", "vector.npy")
        return np.load(path, mmap_mode="r") if os.path.exists(path) else None

    def domains(self):
        """(domain ids [N] int64, domain names) or (None, None)."""
        path = os.path.join(self.root, "labels", "domain.npy")
        if not os.path.exists(path):
            return None, None
        return np.load(path), (self.manifest.domain_names or [])
