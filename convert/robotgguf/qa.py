"""labels-qa — C2's label quality gates (extraction-v1 §3.4), lockfile-writing.

Always runs (pure numpy):
  Q1  per-axis variance per stratum — an axis with no variation inside any
      stratum is UNTESTABLE and flagged before any probe trains
  Q2  tier coverage — which axes are still at t0 (heuristic) vs t1/t2

With ROBOT_TEACHER_* set (and the annotation cache warm), additionally:
  Q3  teacher test-retest correlation on a holdout sample (axes the teacher
      can't score consistently are findings about the axis, not noise)
"""
from __future__ import annotations

import json
import os

import numpy as np

from .config import Config, Lockfile
from .semvec import SemvecSpec


def run(cfg: Config, retest_sample: int = 2000) -> None:
    if not cfg.semvec:
        raise SystemExit("labels-qa: config names no `semvec:` spec")
    spec = SemvecSpec.load(cfg.resolve(cfg.semvec))

    vec_path = os.path.join(cfg.recordings_dir, "labels", "vector.npy")
    if not os.path.exists(vec_path):
        raise SystemExit("labels-qa: no labels/vector.npy — run `robotgguf relabel` first")
    vec = np.load(vec_path, mmap_mode="r")
    dom_path = os.path.join(cfg.recordings_dir, "labels", "domain.npy")
    dom = np.load(dom_path) if os.path.exists(dom_path) else None

    # Q1 — per-stratum variance
    untestable, low_var = [], []
    named = np.asarray(vec[:, : spec.named_dim], dtype=np.float32)
    for a in spec.axes:
        col = named[:, a.index]
        if dom is not None:
            per = [float(col[dom == d].std()) for d in np.unique(dom)
                   if (dom == d).sum() >= 100]
            worst_ok = max(per) if per else float(col.std())
        else:
            worst_ok = float(col.std())
        if worst_ok <= 1e-6:
            untestable.append(a.name)
        elif worst_ok < 0.1:
            low_var.append(a.name)

    # Q2 — tier coverage
    src_path = os.path.join(cfg.recordings_dir, "labels", "vector_sources.json")
    sources = {}
    if os.path.exists(src_path):
        with open(src_path) as f:
            sources = json.load(f).get("sources") or {}
    by_tier = {"t0": 0, "t1": 0, "t2": 0}
    for a in spec.axes:
        t = sources.get(a.name)
        if t in by_tier:
            by_tier[t] += 1
    unlabeled = [a.name for a in spec.axes if a.name not in sources]

    report = {"untestable": untestable, "low_variance": low_var,
              "tier_coverage": by_tier, "unlabeled": unlabeled[:32],
              "latent": "present" if sources.get("__latent__") else "absent"}
    print(f"labels-qa: {len(untestable)} untestable axis(es), "
          f"{len(low_var)} low-variance; coverage {by_tier}, "
          f"{len(unlabeled)} axis(es) unlabeled; latent {report['latent']}")

    # Q3 — teacher test-retest (only when an endpoint is configured)
    if os.environ.get("ROBOT_TEACHER_BASE_URL"):
        from . import labelers_teacher  # noqa: PLC0415
        cache_dir = cfg.workdir
        sents_path = os.path.join(cfg.recordings_dir, "labels", "qa-holdout.json")
        if os.path.exists(sents_path):
            with open(sents_path) as f:
                sents = json.load(f)[:retest_sample]
            ca = labelers_teacher.AnnotationCache(
                os.path.join(cache_dir, "teacher-cache.jsonl"), spec)
            cb = labelers_teacher.AnnotationCache(
                os.path.join(cache_dir, "teacher-cache-retest.jsonl"), spec)
            report["test_retest"] = labelers_teacher.test_retest(spec, sents, ca, cb)
            weak = [k for k, v in report["test_retest"].items()
                    if v is not None and v < 0.6]
            print(f"labels-qa: test-retest on {len(sents)} sentence(s); "
                  f"{len(weak)} axis(es) below r=0.6: {weak[:8]}")
        else:
            print("labels-qa: no labels/qa-holdout.json — run `robotgguf labelvec` "
                  "first (it samples the holdout)")
    else:
        print("labels-qa: teacher not configured (ROBOT_TEACHER_BASE_URL) — "
              "test-retest skipped")

    Lockfile(cfg.lockfile_path).update("labels_qa", report)
    print("labels-qa: report → lockfile [labels_qa]")
