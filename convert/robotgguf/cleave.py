"""R2 — Cleave: discover typed bottlenecks in the frozen donor (§D).

For each candidate site × attribute, train a small linear probe on the
recordings and score:

  decodability — held-out accuracy
  selectivity  — margin over the same probe trained on a random equal-width
                 slice at the same layer (approximated with a shuffled-feature
                 control when no sibling slice was recorded)
  stability    — 1 − accuracy std across corpus shards

Sites clear the config's bars for at least one attribute → bottleneck entries
+ admission scores land in the lockfile, and the winning probes are kept as
extension-tensor material for R7 (identity-checkable by the runtime's
llama_robot_probe_eval). No core fine-tuning: attributes that aren't decodable
in the frozen donor are dropped and recorded as findings.
"""
from __future__ import annotations

import os

import numpy as np

from .config import Config, Lockfile
from .recordings import RecordingStore


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _balanced_accuracy(pred: np.ndarray, y: np.ndarray, classes: int) -> float:
    """Mean per-class recall — chance level is 1/C regardless of imbalance, so
    majority-class guessing scores ~1/C instead of the majority fraction. This
    is the metric that makes decodability meaningful on real (imbalanced) web
    corpora, where raw accuracy just rewards predicting the dominant class."""
    recalls = []
    for c in range(classes):
        m = (y == c)
        if m.any():
            recalls.append(float((pred[m] == c).mean()))
    return float(np.mean(recalls)) if recalls else 0.0


def train_probe(x: np.ndarray, y: np.ndarray, l2: float = 1e-3,
                epochs: int = 200, lr: float = 0.5, seed: int = 0):
    """Class-weighted multinomial logistic regression by full-batch gradient
    descent. Returns (W [D, C], b [C], held-out balanced accuracy). Inverse-
    frequency class weights keep the probe from ignoring minority classes; the
    score is balanced accuracy so imbalance can't inflate it."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float32)
    n, d = x.shape
    classes = int(y.max()) + 1

    # standardize (folded back into W/b so exported probes act on raw slices)
    mu, sd = x.mean(0), x.std(0) + 1e-6
    xs = (x - mu) / sd

    idx = rng.permutation(n)
    split = max(1, int(0.8 * n))
    tr, te = idx[:split], idx[split:]
    ytr = y[tr]

    # inverse-frequency per-sample weights (normalized to mean 1)
    counts = np.bincount(ytr, minlength=classes).astype(np.float32)
    cls_w = len(ytr) / (classes * np.maximum(counts, 1.0))
    sw = cls_w[ytr]
    sw = (sw / sw.mean()).astype(np.float32)

    w = np.zeros((d, classes), dtype=np.float32)
    b = np.zeros(classes, dtype=np.float32)
    onehot = np.eye(classes, dtype=np.float32)[ytr]
    for _ in range(epochs):
        p = _softmax(xs[tr] @ w + b)
        err = (p - onehot) * sw[:, None]              # weight the residual
        g = xs[tr].T @ err / len(tr) + l2 * w
        w -= lr * g
        b -= lr * err.mean(0)

    if len(te):
        pred = np.argmax(xs[te] @ w + b, axis=1)
        bacc = _balanced_accuracy(pred, y[te], classes)
    else:
        bacc = 0.0

    # unfold standardization: probe(raw) = ((raw − mu)/sd)·W + b
    w_raw = (w / sd[:, None]).astype(np.float32)
    b_raw = (b - mu / sd @ w).astype(np.float32)
    return w_raw, b_raw, bacc


def _shard_accuracy(x, y, w, b, shards) -> list:
    accs = []
    classes = int(np.asarray(y).max()) + 1
    for lo, hi in zip(shards[:-1], shards[1:]):
        if hi > lo:
            pred = np.argmax(np.asarray(x[lo:hi], dtype=np.float32) @ w + b, axis=1)
            accs.append(_balanced_accuracy(pred, np.asarray(y[lo:hi]), classes))
    return accs


def run(cfg: Config) -> None:
    store = RecordingStore(cfg.recordings_dir)
    if not store.exists():
        raise SystemExit(f"no recordings at {cfg.recordings_dir} — run `robotgguf record` first")
    man = store.manifest
    lock = Lockfile(cfg.lockfile_path)
    rng = np.random.default_rng(1)

    probe_dir = os.path.join(cfg.workdir, "probes")
    os.makedirs(probe_dir, exist_ok=True)

    import sys, time  # noqa: PLC0415
    bottlenecks, findings = [], []
    n_sites = len(man.sites)
    n_pairs = n_sites * len(man.attributes)
    print(f"cleave: training {n_pairs} probe(s) + {n_pairs} selectivity controls "
          f"over {man.n_samples:,} samples", file=sys.stderr, flush=True)
    t0 = time.time()
    done = 0
    for si, (name, site) in enumerate(man.sites.items()):
        x = store.activations(name)
        admitted_attrs, scores = [], {}
        for attr in man.attributes:
            y = store.labels(attr)
            w, b, acc = train_probe(x, y)

            # selectivity control: same probe capacity on a class-decorrelated
            # (row-shuffled) copy of the slice
            perm = rng.permutation(len(y))
            _, _, acc_ctl = train_probe(np.asarray(x)[perm], y)
            sel = acc - acc_ctl

            stab = 1.0 - float(np.std(_shard_accuracy(x, y, w, b, man.shards)))

            done += 1
            el = time.time() - t0
            eta = el / done * (n_pairs - done)
            verdict = "keep" if (acc >= cfg.min_decodability and sel >= cfg.min_selectivity) else "drop"
            print(f"\rcleave: [{done}/{n_pairs}] {name}.{attr:16} "
                  f"decod={acc:.3f} sel={sel:+.3f} → {verdict}   ETA {eta:.0f}s   ",
                  end="", file=sys.stderr, flush=True)

            if acc >= cfg.min_decodability and sel >= cfg.min_selectivity:
                admitted_attrs.append(attr)
                scores[attr] = {"decodability": round(acc, 4),
                                "selectivity": round(sel, 4),
                                "stability": round(stab, 4)}
                np.save(os.path.join(probe_dir, f"{name}.{attr}.weight.npy"), w.T)  # [C, D] rows
                np.save(os.path.join(probe_dir, f"{name}.{attr}.bias.npy"), b)
            else:
                findings.append({"site": name, "attribute": attr,
                                 "decodability": round(acc, 4),
                                 "selectivity": round(sel, 4),
                                 "verdict": "not decodable in the frozen donor — dropped"})

        if admitted_attrs:
            bn = dict(site)
            bn["name"] = name
            bn["attributes"] = admitted_attrs
            bn["decodability"] = max(s["decodability"] for s in scores.values())
            bn["selectivity"] = max(s["selectivity"] for s in scores.values())
            bn["scores"] = scores
            bottlenecks.append(bn)

    bottlenecks.sort(key=lambda b: -b["decodability"])
    bottlenecks = bottlenecks[: cfg.max_bottlenecks]

    lock.update("cleave", {
        "recordings": {"model": man.model, "corpus": man.corpus},
        "bottlenecks": bottlenecks,
        "findings": findings,
        "probe_dir": probe_dir,  # absolute; round-trips through cfg.resolve
    })
    print(f"cleave: admitted {len(bottlenecks)} bottleneck(s), "
          f"{len(findings)} attribute/site pair(s) dropped as findings")
    for bn in bottlenecks:
        print(f"  {bn['name']}: layer {bn['layer']} {bn['point']} "
              f"[{bn['offset']}..{bn['offset'] + bn['width']}) "
              f"attrs={bn['attributes']} decod={bn['decodability']}")

    # extraction-v1: the vector path (ridge map into semvec) runs whenever the
    # recordings carry labels/vector.npy — same lockfile, its own section
    from . import cleave_vec  # noqa: PLC0415
    cleave_vec.run(cfg)
