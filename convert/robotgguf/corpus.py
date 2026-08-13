"""Corpus manifest + stratified token loading (extraction-v1 §2, package C1).

v0 read `cfg.corpus` files head-to-tail and stopped at the token budget, so
recordings drew from the head of the first file (extraction-v1 §1.5 — the
loader bug). v1 interleaves *strata* by share: every stratum is present in
every recording at its configured proportion, regardless of budget, and every
token window carries a domain id (the stratification key for cross-domain
admission).

The manifest (`corpus/manifest.yaml`) is written by tools/fetch_corpus.py and
consumed here:

    strata:
      - { domain: web-en, file: corpus/web-en.txt, share: 0.25 }
      ...
    provenance:
      - { domain: web-en, dataset: HuggingFaceFW/fineweb, license: odc-by }

Back-compat: a bare `corpus:` file list (v0 configs) becomes one stratum per
file with equal shares — which *changes* v0's effective sampling (that is the
fix, not a regression; re-baseline per extraction-v1 C1).
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field

import numpy as np
import yaml


@dataclass
class Stratum:
    domain: str
    file: str
    share: float
    meta: dict = field(default_factory=dict)


def load_strata(cfg) -> list:
    """Resolve cfg.corpus into a list of Stratum.

    Accepts either a single manifest path (str or 1-list ending in .yaml/.yml)
    or the v0 bare list of text files.
    """
    spec = cfg.corpus
    if isinstance(spec, str):
        spec = [spec]
    if not spec:
        raise SystemExit("corpus: config lists no corpus files or manifest")

    if len(spec) == 1 and str(spec[0]).endswith((".yaml", ".yml")):
        path = cfg.resolve(spec[0])
        with open(path) as f:
            man = yaml.safe_load(f) or {}
        strata = []
        base = os.path.dirname(path)
        for s in man.get("strata") or []:
            fp = s["file"]
            if not os.path.isabs(fp):
                # paths in the manifest are authored relative to the convert
                # root (like every other config path); fall back to
                # manifest-dir-relative for hand-written manifests
                cand = cfg.resolve(fp)
                fp = cand if os.path.exists(cand) else \
                    os.path.normpath(os.path.join(base, os.path.basename(fp)))
            strata.append(Stratum(domain=s["domain"], file=fp,
                                  share=float(s["share"]), meta=s.get("meta") or {}))
        if not strata:
            raise SystemExit(f"corpus: manifest {path} has no strata")
        total = sum(s.share for s in strata)
        for s in strata:
            s.share /= total
        return strata

    # v0 bare list → one stratum per file, equal shares
    files = [cfg.resolve(p) for p in spec]
    n = len(files)
    return [Stratum(domain=os.path.splitext(os.path.basename(p))[0],
                    file=p, share=1.0 / n) for p in files]


def strata_hash(strata: list) -> str:
    h = hashlib.sha256()
    for s in strata:
        h.update(f"{s.domain}:{s.share:.6f}:{os.path.basename(s.file)}".encode())
    return h.hexdigest()[:16]


class _StratumReader:
    """Chunked text→token reader for one stratum. Tokenizes ~1 MB of text at
    a time and hands out exactly `window`-sized id chunks."""

    def __init__(self, stratum: Stratum, tok, window: int):
        self.stratum = stratum
        self.tok = tok
        self.window = window
        self._fh = open(stratum.file, encoding="utf-8", errors="ignore")
        self._ids: list = []
        self._eof = False

    def _refill(self) -> None:
        buf, buf_len = [], 0
        for line in self._fh:
            buf.append(line)
            buf_len += len(line)
            if buf_len >= 1_000_000:
                break
        if not buf:
            self._eof = True
            return
        self._ids.extend(self.tok("".join(buf)).input_ids)

    def next_window(self):
        """Return the next `window` token ids, or None at EOF."""
        while len(self._ids) < self.window and not self._eof:
            self._refill()
        if len(self._ids) < self.window:
            return None
        out, self._ids = self._ids[: self.window], self._ids[self.window:]
        return out

    def close(self) -> None:
        self._fh.close()


def stratified_windows(strata: list, tok, max_tokens: int, window: int,
                       seed: int = 20260708):
    """Interleave whole windows across strata in proportion to share.

    Returns (windows, domain_ids, domains):
      windows    — list of token-id lists, each exactly `window` long
      domain_ids — int64 array [len(windows)], index into `domains`
      domains    — list of domain names (stable order = manifest order)

    Deterministic for a given (strata, seed): scheduling is deficit-based
    (largest share-weighted deficit fires next; the rng only tie-breaks), so
    survey and focused passes sample identically (extraction-v1 §8 drift
    risk). A stratum that runs out of text is dropped and the remaining
    shares renormalize, with a warning — silent share drift would poison the
    per-domain admission bars.
    """
    import sys

    rng = np.random.default_rng(seed)
    readers = [_StratumReader(s, tok, window) for s in strata]
    domains = [s.domain for s in strata]
    n_windows_target = max_tokens // window
    if n_windows_target == 0:
        raise SystemExit(f"corpus: max_tokens {max_tokens} < window {window}")

    emitted = np.zeros(len(readers))          # windows emitted per stratum
    shares = np.array([s.share for s in strata], dtype=np.float64)
    alive = np.ones(len(readers), dtype=bool)

    windows, domain_ids = [], []
    while len(windows) < n_windows_target and alive.any():
        sh = shares * alive
        sh = sh / sh.sum()
        deficit = sh * (len(windows) + 1) - emitted
        deficit[~alive] = -np.inf
        top = np.flatnonzero(deficit >= deficit.max() - 1e-12)
        pick = int(top[0]) if len(top) == 1 else int(rng.choice(top))

        got = readers[pick].next_window()
        if got is None:
            alive[pick] = False
            print(f"corpus: stratum '{domains[pick]}' exhausted after "
                  f"{int(emitted[pick])} window(s) — renormalizing shares",
                  file=sys.stderr, flush=True)
            continue
        windows.append(got)
        domain_ids.append(pick)
        emitted[pick] += 1

    for r in readers:
        r.close()
    if not windows:
        raise SystemExit("corpus: no stratum produced a full window of tokens")
    return windows, np.asarray(domain_ids, dtype=np.int64), domains


def domain_shares_report(domain_ids: np.ndarray, domains: list) -> dict:
    n = len(domain_ids)
    return {d: round(float((domain_ids == i).sum()) / n, 4)
            for i, d in enumerate(domains)}
