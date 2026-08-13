"""T2 — teacher-LLM axis scoring, sampled + distilled (extraction-v1 §3.3).

The FineWeb-Edu / WebOrganizer recipe, widened to a vector: the teacher
scores a stratified SAMPLE of sentences on all named judgment axes in one
structured response per sentence; annotations are cached by (sentence hash,
semvec hash) so nothing is ever paid for twice; a distilled multi-output
ridge head over Block-B embeddings then scores the FULL corpus. The teacher
never sees the whole corpus.

The client speaks the OpenAI-compatible chat API, so the teacher can be a
Modal-hosted vLLM (Qwen3.6-35B-A3B serves both as primary donor and as its
own teacher), a local server, or a commercial endpoint:

    export ROBOT_TEACHER_BASE_URL=https://<modal-app>.modal.run/v1
    export ROBOT_TEACHER_MODEL=Qwen/Qwen3.6-35B-A3B
    export ROBOT_TEACHER_API_KEY=...           # if the endpoint wants one

No third-party HTTP deps — urllib only.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.request

import numpy as np

from .semvec import SemvecSpec

# groups the teacher owns (spec tiers t2) — structural groups are never asked
JUDGMENT_GROUPS = ("affect", "register", "discourse", "epistemics", "safety")


def _endpoint():
    base = os.environ.get("ROBOT_TEACHER_BASE_URL")
    model = os.environ.get("ROBOT_TEACHER_MODEL")
    if not base or not model:
        raise SystemExit("t2: set ROBOT_TEACHER_BASE_URL and ROBOT_TEACHER_MODEL "
                         "(e.g. a Modal vLLM app serving Qwen3.6-35B-A3B)")
    return base.rstrip("/"), model, os.environ.get("ROBOT_TEACHER_API_KEY", "")


def _axes_for_teacher(spec: SemvecSpec) -> list:
    return [a for a in spec.axes if a.group in JUDGMENT_GROUPS]


def _prompt(spec: SemvecSpec, axes: list, sentences: list) -> str:
    lo, hi = spec.scale
    axis_list = "\n".join(f"- {a.name} ({a.group})" for a in axes)
    numbered = "\n".join(f"{i}: {s[:400]}" for i, s in enumerate(sentences))
    return (
        f"Score each numbered text on every axis below, {lo:g}..{hi:g} "
        f"(0 = absent, {hi:g} = extreme). Judge only the text itself.\n"
        f"Axes:\n{axis_list}\n\n"
        f"Texts:\n{numbered}\n\n"
        "Reply with ONLY a JSON object: {\"<index>\": {\"<axis>\": <score>, ...}, ...} "
        "covering every index and every axis."
    )


def _chat(base: str, model: str, key: str, prompt: str,
          temperature: float = 0.0, timeout: int = 240) -> str:
    req = urllib.request.Request(
        base + "/chat/completions",
        data=json.dumps({"model": model, "temperature": temperature,
                         "response_format": {"type": "json_object"},
                         "messages": [{"role": "user", "content": prompt}]}).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {key}"} if key else {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


class AnnotationCache:
    """Append-only JSONL keyed by (sentence sha256, semvec hash). Label once,
    reuse forever — across corpora AND donors (labels are text functions)."""

    def __init__(self, path: str, spec: SemvecSpec):
        self.path = path
        self.key_suffix = spec.spec_hash()
        self._mem: dict = {}
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    row = json.loads(line)
                    self._mem[row["k"]] = row["scores"]

    def key(self, sentence: str) -> str:
        return hashlib.sha256(sentence.encode()).hexdigest()[:24] + ":" + self.key_suffix

    def get(self, sentence: str):
        return self._mem.get(self.key(sentence))

    def put(self, sentence: str, scores: dict) -> None:
        k = self.key(sentence)
        self._mem[k] = scores
        with open(self.path, "a") as f:
            f.write(json.dumps({"k": k, "scores": scores}) + "\n")


def annotate(spec: SemvecSpec, sentences: list, cache: AnnotationCache,
             batch: int = 8, temperature: float = 0.0) -> np.ndarray:
    """Teacher-score `sentences` on the judgment axes. Returns [N, n_axes]
    float32 with NaN where the teacher failed. Cache-first."""
    base, model, key = _endpoint()
    axes = _axes_for_teacher(spec)
    out = np.full((len(sentences), len(axes)), np.nan, dtype=np.float32)

    todo = [i for i, s in enumerate(sentences) if cache.get(s) is None]
    print(f"t2: {len(sentences) - len(todo)} cached, {len(todo)} to annotate",
          file=sys.stderr, flush=True)
    for lo in range(0, len(todo), batch):
        idx = todo[lo:lo + batch]
        try:
            raw = _chat(base, model, key, _prompt(spec, axes, [sentences[i] for i in idx]),
                        temperature=temperature)
            got = json.loads(raw)
        except Exception as e:  # noqa: BLE001 — a failed batch is NaN, not fatal
            print(f"t2: batch @{lo} failed ({e}) — leaving NaN", file=sys.stderr)
            continue
        for bi, i in enumerate(idx):
            scores = got.get(str(bi))
            if scores:                     # don't cache misses — retryable next run
                cache.put(sentences[i], scores)
        if (lo // batch) % 20 == 0:
            print(f"t2: {lo + len(idx)}/{len(todo)}", file=sys.stderr, flush=True)

    lo_s, hi_s = spec.scale
    for i, s in enumerate(sentences):
        scores = cache.get(s)
        if scores:
            for j, a in enumerate(axes):
                v = scores.get(a.name)
                if v is not None:
                    out[i, j] = min(max(float(v), lo_s), hi_s)
    return out


def distill(teacher_scores: np.ndarray, embeddings: np.ndarray, l2: float = 1.0):
    """Multi-output ridge: Block-B-space embeddings → judgment axes. Returns
    (W [E+1, A], per-axis held-out Pearson r). Axes whose r misses the C2 bar
    are the caller's to demote (teacher-sample-only), never to force."""
    x = np.asarray(embeddings, dtype=np.float64)
    y = np.asarray(teacher_scores, dtype=np.float64)
    keep = ~np.isnan(y).any(axis=1)
    x, y = x[keep], y[keep]
    n = len(x)
    if n < 200:
        raise SystemExit(f"t2: only {n} fully-scored sentences — annotate more before distilling")
    xb = np.concatenate([x, np.ones((n, 1))], axis=1)
    split = int(0.8 * n)
    a = xb[:split].T @ xb[:split] + l2 * np.eye(xb.shape[1])
    w = np.linalg.solve(a, xb[:split].T @ y[:split])
    pred = xb[split:] @ w
    r = np.array([_pearson(pred[:, j], y[split:, j]) for j in range(y.shape[1])])
    return w.astype(np.float32), r


def test_retest(spec: SemvecSpec, sentences: list, cache_a: AnnotationCache,
                cache_b: AnnotationCache) -> dict:
    """C2 QA: score the holdout twice (different caches/temperatures), report
    per-axis correlation. Axes the teacher can't score consistently are
    findings about the axis."""
    a = annotate(spec, sentences, cache_a, temperature=0.0)
    b = annotate(spec, sentences, cache_b, temperature=0.7)
    axes = _axes_for_teacher(spec)
    out = {}
    for j, ax in enumerate(axes):
        m = ~(np.isnan(a[:, j]) | np.isnan(b[:, j]))
        out[ax.name] = round(_pearson(a[m, j], b[m, j]), 3) if m.sum() >= 30 else None
    return out


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or x.std() < 1e-9 or y.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])
