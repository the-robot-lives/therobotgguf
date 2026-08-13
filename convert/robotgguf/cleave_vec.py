"""R2 v1 — vector cleave: one ridge map per site into semvec
(extraction-v1 §4). Runs alongside the v0 categorical path whenever the
recording store carries labels/vector.npy.

Per site X ∈ ℝ^{N×d}: standardize, closed-form ridge to the label vector
Y ∈ ℝ^{N×D}, then per-axis scores on the held-out split:

  decodability      — Spearman ρ (named/ordinal axes) or Pearson r (latent)
  selectivity       — decodability − same-capacity map on row-shuffled X
  stability         — 1 − std across the manifest's shards
  domain_stability  — min per-stratum held-out decodability (labels/domain.npy)

Axes clearing the config bars are ADMITTED; the site's exported material is
the admitted rows of the raw-space map (proj [d, D], unadmitted columns
zeroed) plus per-axis calibration (bias from the standardization unfold) —
exactly what R7 packages as `robot.semvec.{site}.proj/.calib` and the
runtime evaluates as one matvec (extraction-v1 §4.4). Named axes that miss
the bar linearly get one MLP retry; an MLP-only admission is recorded as a
finding (nonlinearly present), never exported as a linear probe.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

from .config import Config, Lockfile
from .recordings import RecordingStore
from .semvec import SemvecSpec


# ---- scoring ----

def _rank(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, axis=0, kind="stable")
    r = np.empty_like(order, dtype=np.float64)
    np.put_along_axis(r, order, np.arange(len(a), dtype=np.float64)[:, None], axis=0)
    return r


def _pearson_cols(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = p - p.mean(0)
    y = y - y.mean(0)
    denom = np.sqrt((p * p).sum(0) * (y * y).sum(0)) + 1e-12
    return (p * y).sum(0) / denom


def axis_scores(pred: np.ndarray, y: np.ndarray, named_dim: int) -> np.ndarray:
    """Per-axis decodability: Spearman ρ for named axes, Pearson r for the
    latent block. Zero-variance axes score 0 (untestable, not failing)."""
    out = np.zeros(y.shape[1])
    var = y.std(0) > 1e-6
    named = slice(0, named_dim)
    lat = slice(named_dim, y.shape[1])
    if named_dim:
        s = _pearson_cols(_rank(pred[:, named]), _rank(y[:, named]))
        out[named] = s
    if y.shape[1] > named_dim:
        out[lat] = _pearson_cols(pred[:, lat], y[:, lat])
    out[~var] = 0.0
    return out


# ---- the ridge map ----

def fit_ridge(x: np.ndarray, y: np.ndarray, l2: float = 10.0):
    """Standardized ridge, closed form. Returns (W_std [d, D], mu, sd)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mu, sd = x.mean(0), x.std(0) + 1e-6
    xs = (x - mu) / sd
    d = xs.shape[1]
    w = np.linalg.solve(xs.T @ xs + l2 * np.eye(d), xs.T @ y)
    return w, mu, sd


def unfold(w: np.ndarray, mu: np.ndarray, sd: np.ndarray):
    """Fold standardization back so the exported map acts on RAW slices:
    y = raw @ W_raw + b  ≡  ((raw − mu)/sd) @ W."""
    w_raw = w / sd[:, None]
    b_raw = -(mu / sd) @ w
    return w_raw.astype(np.float32), b_raw.astype(np.float32)


def _mlp_score(x_tr, y_tr, x_te, y_te, named: bool, hidden: int = 64,
               epochs: int = 400, lr: float = 0.5, seed: int = 3) -> float:
    """One small numpy MLP (the nonlinear fallback) → held-out decodability
    for a single axis. Regression head, tanh hidden layer."""
    rng = np.random.default_rng(seed)
    d = x_tr.shape[1]
    w1 = rng.normal(0, 1.0 / np.sqrt(d), (d, hidden))
    b1 = np.zeros(hidden)
    w2 = rng.normal(0, 1.0 / np.sqrt(hidden), hidden)
    b2 = 0.0
    y_mu, y_sd = y_tr.mean(), y_tr.std() + 1e-6
    yt = (y_tr - y_mu) / y_sd
    for _ in range(epochs):
        h = np.tanh(x_tr @ w1 + b1)
        p = h @ w2 + b2
        g = (p - yt) / len(yt)
        w2g = h.T @ g
        hg = np.outer(g, w2) * (1 - h * h)
        w1 -= lr * x_tr.T @ hg
        b1 -= lr * hg.sum(0)
        w2 -= lr * w2g
        b2 -= lr * g.sum()
    p_te = np.tanh(x_te @ w1 + b1) @ w2 + b2
    if named:
        return float(_pearson_cols(_rank(p_te[:, None]), _rank(y_te[:, None]))[0])
    return float(_pearson_cols(p_te[:, None], y_te[:, None])[0])


# ---- the run ----

def run(cfg: Config) -> None:
    store = RecordingStore(cfg.recordings_dir)
    vec = store.label_vector()
    if vec is None:
        print("cleave-vec: no labels/vector.npy — v0 categorical path only", file=sys.stderr)
        return
    if not cfg.semvec:
        raise SystemExit("cleave-vec: recordings carry a label vector but the config "
                         "names no `semvec:` spec")
    spec = SemvecSpec.load(cfg.resolve(cfg.semvec))
    man = store.manifest
    if man.semvec and man.semvec.get("hash") not in (None, spec.spec_hash()):
        raise SystemExit(f"cleave-vec: recordings were labeled under semvec hash "
                         f"{man.semvec.get('hash')} but the spec resolves to "
                         f"{spec.spec_hash()} — coordinate systems must match")

    lock = Lockfile(cfg.lockfile_path)
    rng = np.random.default_rng(1)
    dom_ids, dom_names = store.domains()

    # sample cap keeps the solve + scoring memory-bounded on focused passes
    n_all = min(man.n_samples, len(vec))   # vector may be truncated vs acts
    cap = int(cfg.vec_sample_cap)
    take = np.sort(rng.choice(n_all, size=min(cap, n_all), replace=False))
    y_all = np.asarray(vec, dtype=np.float32)[take]
    d_ids = dom_ids[take] if dom_ids is not None else None

    perm = rng.permutation(len(take))
    split = max(1, int(0.8 * len(take)))
    tr, te = perm[:split], perm[split:]

    out_dir = os.path.join(cfg.workdir, "semvec-probes")
    os.makedirs(out_dir, exist_ok=True)
    names = [a.name for a in spec.axes] + \
            [f"latent_{i}" for i in range(spec.latent_dim)]

    results, findings = {}, []
    t0 = time.time()
    for si, (sname, site) in enumerate(man.sites.items()):
        x = np.asarray(store.activations(sname), dtype=np.float32)[take]

        w, mu, sd = fit_ridge(x[tr], y_all[tr], l2=float(cfg.vec_l2))
        xs_te = (x[te] - mu) / sd
        pred = xs_te @ w
        dec = axis_scores(pred, y_all[te], spec.named_dim)

        # selectivity control: same capacity on row-shuffled X
        sh = rng.permutation(len(tr))
        w_c, mu_c, sd_c = fit_ridge(x[tr][sh], y_all[tr], l2=float(cfg.vec_l2))
        dec_c = axis_scores(((x[te] - mu_c) / sd_c) @ w_c, y_all[te], spec.named_dim)
        sel = dec - dec_c

        # stability across manifest shards (positional, as v0)
        shard_scores = []
        for lo, hi in zip(man.shards[:-1], man.shards[1:]):
            m = (take[te] >= lo) & (take[te] < hi)
            if m.sum() >= 50:
                shard_scores.append(axis_scores(pred[m], y_all[te][m], spec.named_dim))
        stab = 1.0 - (np.std(shard_scores, axis=0) if len(shard_scores) >= 2
                      else np.zeros(spec.dim))

        # domain stability: min per-stratum held-out decodability, masked to
        # strata where the axis actually varies (extraction-v1 §4.3)
        dom_min = np.full(spec.dim, np.nan)
        if d_ids is not None and dom_names:
            per_dom = []
            for di in range(len(dom_names)):
                m = d_ids[te] == di
                if m.sum() >= 50:
                    s = axis_scores(pred[m], y_all[te][m], spec.named_dim)
                    s[y_all[te][m].std(0) <= 1e-6] = np.nan   # absent from stratum
                    per_dom.append(s)
            if per_dom:
                import warnings  # noqa: PLC0415
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN axes
                    dom_min = np.nanmin(per_dom, axis=0)

        min_dec = float(cfg.min_axis_decodability)
        min_sel = float(cfg.min_axis_selectivity)
        dom_bar = float(cfg.min_domain_stability_ratio) * min_dec
        admit = (dec >= min_dec) & (sel >= min_sel) & \
                (np.isnan(dom_min) | (dom_min >= dom_bar))

        # MLP fallback for named axes that miss linearly (capped)
        retried = 0
        for j in np.flatnonzero(~admit[: spec.named_dim]):
            if retried >= int(cfg.mlp_fallback_max):
                break
            if y_all[:, j].std() <= 1e-6 or dec[j] >= min_dec:
                continue
            retried += 1
            mscore = _mlp_score(( x[tr] - mu) / sd, y_all[tr][:, j],
                                xs_te, y_all[te][:, j], named=True)
            if mscore >= min_dec:
                findings.append({"site": sname, "axis": names[j],
                                 "linear": round(float(dec[j]), 3),
                                 "mlp": round(mscore, 3),
                                 "verdict": "nonlinearly present — not exportable as a linear probe"})

        w_raw, b_raw = unfold(w, mu, sd)
        w_raw[:, ~admit] = 0.0
        calib = np.zeros((spec.dim, 2), dtype=np.float32)
        calib[admit, 0] = 1.0
        calib[admit, 1] = b_raw[admit]
        np.save(os.path.join(out_dir, f"{sname}.proj.npy"), w_raw)
        np.save(os.path.join(out_dir, f"{sname}.calib.npy"), calib)

        # the write path: semvec → slice overlay, write-calibrated against
        # this site's own encoder (overlay.py — extraction-v1 §4.4/§4.5)
        from . import overlay as overlay_mod  # noqa: PLC0415
        g_fit, _ = overlay_mod.fit_overlay(y_all[tr], x[tr], l2=float(cfg.vec_l2))
        g_cal, writable, crosstalk = overlay_mod.calibrate(g_fit, w_raw, admit)
        np.save(os.path.join(out_dir, f"{sname}.overlay.npy"), g_cal)

        admitted = [{"axis": names[j], "decodability": round(float(dec[j]), 4),
                     "selectivity": round(float(sel[j]), 4),
                     "stability": round(float(stab[j]), 4),
                     "writable": bool(writable[j]),
                     "write_crosstalk": round(float(crosstalk[j]), 4),
                     **({} if np.isnan(dom_min[j]) else
                        {"domain_stability": round(float(dom_min[j]), 4)})}
                    for j in np.flatnonzero(admit)]
        results[sname] = {"n_admitted": int(admit.sum()),
                          "n_admitted_named": int(admit[: spec.named_dim].sum()),
                          "n_writable": int(writable.sum()),
                          "axes": admitted}
        el = time.time() - t0
        print(f"cleave-vec: [{si + 1}/{len(man.sites)}] {sname}: "
              f"{int(admit[: spec.named_dim].sum())}/{spec.named_dim} named + "
              f"{int(admit[spec.named_dim:].sum())}/{spec.latent_dim} latent admitted "
              f"({el:.0f}s)", file=sys.stderr, flush=True)

    lock.update("cleave_vec", {
        "semvec": {"version": spec.version, "hash": spec.spec_hash()},
        "sample_cap": int(min(cap, n_all)),
        "bars": {"min_axis_decodability": float(cfg.min_axis_decodability),
                 "min_axis_selectivity": float(cfg.min_axis_selectivity),
                 "min_domain_stability_ratio": float(cfg.min_domain_stability_ratio)},
        "probe_dir": out_dir,
        "sites": results,
        "findings": findings,
    })
    print(f"cleave-vec: {sum(r['n_admitted'] for r in results.values())} axis-site "
          f"admission(s) across {len(results)} site(s); {len(findings)} nonlinear finding(s)")
