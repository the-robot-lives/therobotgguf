"""The semvec overlay — the write path of the standardized layer
(extraction-v1 §4.4/§4.5; the read path is cleave_vec's proj).

Per admitted site, cleave-vec exports an encoder/decoder pair:

  proj    E : ℝ^d → ℝ^D   read  — s = h_slice · E + b      (one matvec)
  overlay G : ℝ^D → ℝ^d   write — h_slice' = h_slice + Δs · G

A reusable module lives ENTIRELY in semvec space — it is a function
Δs = f(s) with no donor-specific weights. Compiling it for a donor@site is
just sandwiching it between that site's E and G:

    s  = E(h)                # standardized read
    Δs = module(s)           # portable logic (any nested net, rule, dial)
    h' = h + scale · Δs · G  # standardized overlay onto the stream

Function-preserving by construction: Δs = 0 ⇒ h' = h (the graft invariant).

G is fit from the same recordings as E (ridge from the label vector back to
the centered activation slice) and then WRITE-CALIBRATED per admitted axis:
rows are scaled so that writing +1.0 on axis j moves the site's own readout
of axis j by exactly +1.0 (roundtrip gain E∘G = identity on the admitted
diagonal). Off-diagonal roundtrip terms are the WRITE CROSSTALK — reported
to the lockfile, because a write that moves neighbors is exactly the failure
mode shim admission exists to catch (extraction-v1 §8).
"""
from __future__ import annotations

import numpy as np


def fit_overlay(y_tr: np.ndarray, x_tr: np.ndarray, l2: float = 10.0):
    """Ridge from semvec Y to the centered activation slice X.
    Returns (G [D, d], x_mean [d]) — G maps semvec DELTAS to slice deltas,
    so only the centered fit matters (the mean never re-enters)."""
    y = np.asarray(y_tr, dtype=np.float64)
    x = np.asarray(x_tr, dtype=np.float64)
    xc = x - x.mean(0)
    dd = y.shape[1]
    g = np.linalg.solve(y.T @ y + l2 * np.eye(dd), y.T @ xc)
    return g, x.mean(0)


def calibrate(g: np.ndarray, w_raw: np.ndarray, admit: np.ndarray,
              min_gain: float = 1e-3):
    """Write-calibrate G against the site's own encoder:

      roundtrip[j, k] = (unit write on axis j) → (readout change on axis k)
                      = G[j] @ W_raw[:, k]

    Admitted rows are scaled so roundtrip[j, j] = 1; axes whose raw gain is
    below `min_gain` are write-refused (row zeroed, recorded) — a write the
    site cannot even read back is not a write, it's noise injection.
    Non-admitted rows are zeroed outright.

    Returns (G_cal, writable [D] bool, crosstalk [D] float — max |off-diag|
    roundtrip per writable axis, computed post-calibration over writable
    columns)."""
    g = np.array(g, dtype=np.float64, copy=True)
    d_dim = g.shape[0]
    writable = np.zeros(d_dim, dtype=bool)
    for j in range(d_dim):
        if not admit[j]:
            g[j] = 0.0
            continue
        gain = float(g[j] @ w_raw[:, j])
        if abs(gain) < min_gain:
            g[j] = 0.0
            continue
        g[j] /= gain
        writable[j] = True

    crosstalk = np.zeros(d_dim)
    if writable.any():
        rt = g[writable] @ w_raw[:, writable]        # [Wr, Wr], diag = 1
        off = np.abs(rt - np.eye(rt.shape[0]))
        crosstalk[writable] = off.max(axis=1)
    return g.astype(np.float32), writable, crosstalk


def apply_overlay(h_slice: np.ndarray, delta_s: np.ndarray, g: np.ndarray,
                  scale: float = 1.0) -> np.ndarray:
    """Reference semantics for the runtime op (one matvec + add):
    h' = h + scale · Δs · G. Δs = 0 is exactly the identity."""
    return np.asarray(h_slice) + scale * (np.asarray(delta_s) @ np.asarray(g))


def read(h_slice: np.ndarray, w_raw: np.ndarray, b_raw: np.ndarray) -> np.ndarray:
    """Reference read: s = h · E + b (matches the runtime's semvec_read)."""
    return np.asarray(h_slice) @ np.asarray(w_raw) + np.asarray(b_raw)
