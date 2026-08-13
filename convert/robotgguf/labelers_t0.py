"""T0 — structural semvec axes (extraction-v1 §3.3): free, deterministic,
dependency-light. Fills the structure/entities/language groups exactly, and
pre-fills weak-heuristic scores for a handful of judgment axes (register,
affect, discourse, safety, topic) that T1/T2 later overwrite — the v0
lexicons live on here as the bootstrap tier.

Sentence-granular like labelers.py: score sentences, positions inherit.
All scores land on the spec's ordinal scale (default 0-4).
"""
from __future__ import annotations

import math
import re

import numpy as np

from . import labelers as v0
from .semvec import SemvecSpec, VectorBuilder

_URL = re.compile(r"https?://|www\.")
_MATH = re.compile(r"[=+±×÷≤≥≠∑∏∫√∂∇]|\\(?:frac|sum|int|alpha|beta|cdot|mathbb)|\$[^$]+\$")
_CODEISH = re.compile(r"[{}();]|::|->|=>|==|!=|\bdef\b|\bfn\b|\breturn\b|\bimport\b|</?\w+>")
_LIST_MARK = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s", re.M)
_SELF = {"i", "i'm", "i've", "me", "my", "mine", "we", "our", "us"}
_SECOND = {"you", "your", "you're", "yours", "yourself"}
_TEMPORAL = {"yesterday", "today", "tomorrow", "now", "soon", "later", "ago",
             "monday", "tuesday", "wednesday", "thursday", "friday",
             "saturday", "sunday", "january", "february", "march", "april",
             "may", "june", "july", "august", "september", "october",
             "november", "december", "year", "month", "week"}
_HEDGE = {"maybe", "perhaps", "possibly", "might", "could", "probably",
          "likely", "unlikely", "seems", "appears", "roughly", "about",
          "presumably", "arguably", "somewhat"}


def _sat(x: float, k: float) -> float:
    """Map a non-negative signal to the 0-4 ordinal scale; k = signal level
    that lands at 2.0 (midpoint). Monotone, saturating."""
    return 4.0 * (1.0 - math.exp(-0.6931 * max(x, 0.0) / max(k, 1e-9)))


def score_sentence(spec: SemvecSpec, s: str) -> dict:
    """One sentence → {axis name: 0-4 score} for every T0-computable axis."""
    words = v0._words(s)
    wset = set(words)
    n_ch = max(len(s), 1)
    n_w = max(len(words), 1)
    toks = s.split()
    v0lab = v0.label_sentence(s)

    sym = sum(1 for c in s if not c.isalnum() and not c.isspace())
    dig = sum(c.isdigit() for c in s)
    up = sum(c.isupper() for c in s)
    caps_mid = sum(1 for i, t in enumerate(toks)
                   if i > 0 and t.strip(".,;:!?\"'()")[:1].isupper()
                   and t.strip(".,;:!?\"'()")[1:2].islower())

    out = {
        # ---- structure (exact) ----
        "symbol_density":        _sat(sym / n_ch, 0.08),
        "numeracy":              _sat(dig / n_ch, 0.05),
        "code_ness":             _sat(len(_CODEISH.findall(s)) / n_w, 0.15),
        "math_notation":         _sat(len(_MATH.findall(s)) / n_w, 0.08),
        "indentation_structure": _sat(len(s) - len(s.lstrip()), 4.0),
        "markup_ness":           _sat(s.count("<") + s.count("#") + s.count("*"), 4.0),
        "list_structure":        _sat(len(_LIST_MARK.findall(s)), 1.0),
        "url_presence":          4.0 if _URL.search(s) else 0.0,
        "uppercase_ratio":       _sat(up / n_ch, 0.15),
        "punctuation_density":   _sat(sum(s.count(c) for c in ".,;:!?") / n_w, 0.25),
        "avg_word_length":       _sat(sum(len(w) for w in words) / n_w, 6.0),
        "sentence_length":       _sat(n_w, 25.0),
        "quote_presence":        _sat(s.count('"') + s.count("“") + s.count("'"), 2.0),
        "parenthetical":         _sat(s.count("(") + s.count("["), 1.5),
        "tabular_ness":          _sat(s.count("|") + s.count("\t"), 3.0),
        "whitespace_structure":  _sat(s.count("  ") + s.count("\n"), 3.0),
        # ---- entities (near-exact) ----
        "person_presence":       _sat(caps_mid, 1.5),       # T1/T2 sharpen per-type
        "org_presence":          _sat(caps_mid, 2.5),
        "place_presence":        _sat(caps_mid, 2.5),
        "temporal_reference":    _sat(len(wset & _TEMPORAL), 1.0),
        "self_reference":        _sat(len(wset & _SELF), 1.0),
        "second_person":         _sat(len(wset & _SECOND), 1.0),
        "named_density":         _sat(caps_mid / n_w, 0.12),
        "number_as_entity":      _sat(sum(t.strip(".,").isdigit() for t in toks), 1.5),
        # ---- language (exact, script-level; T1 sharpens to language id) ----
        "latin_script":          0.0, "cjk_script": 0.0,
        "cyrillic_script":       0.0, "other_script": 0.0,
        # ---- weak heuristic pre-fills (T1/T2 overwrite) ----
        "formality":             3.0 if v0lab["register"] == 1 else 1.0,
        "valence":               {0: 1.0, 1: 2.0, 2: 3.0}[v0lab["sentiment"]],
        "question_ness":         4.0 if v0lab["speech_act"] == 0 else 0.0,
        "imperative_ness":       4.0 if v0lab["speech_act"] == 1 else 0.0,
        "physical_threat":       3.0 if v0lab["safety_salience"] == 1 else 0.5,
        "hedging":               _sat(len(wset & _HEDGE), 1.0),
        "tech_software":         _sat(len(wset & v0._TECH), 1.0),
        "science_bio":           _sat(len(wset & v0._SCIENCE), 1.0),
        "politics":              _sat(len(wset & v0._NEWS), 1.0),
    }
    out[["latin_script", "cjk_script", "cyrillic_script",
         "other_script"][v0._char_class(s)]] = 4.0
    # only emit axes the spec actually defines (forward-compat with minors)
    return {k: v for k, v in out.items() if k in spec.by_name}


def fill(builder: VectorBuilder, spec: SemvecSpec, pieces_per_window: list) -> None:
    """Score every sentence, spread to positions (same span walk as
    labelers.label_pieces), write into the builder as tier t0."""
    total = sum(len(p) for p in pieces_per_window)
    per_axis = {name: np.zeros(total, dtype=np.float32)
                for name in _axes_cache(spec)}

    pos = 0
    for pieces in pieces_per_window:
        text = "".join(pieces)
        spans = _sentence_spans(text)
        scored = [score_sentence(spec, text[a:b]) for a, b in spans]
        ci, si = 0, 0
        for piece in pieces:
            mid = ci + max(1, len(piece)) // 2
            while si + 1 < len(spans) and mid >= spans[si][1]:
                si += 1
            for name, val in scored[si].items():
                per_axis[name][pos] = val
            ci += len(piece)
            pos += 1

    for name, vals in per_axis.items():
        builder.set_axis(name, vals, tier="t0")


def _axes_cache(spec: SemvecSpec):
    # every axis T0 *might* emit — union of score_sentence's keys ∩ spec
    probe = score_sentence(spec, "Probe sentence, quite plain.")
    extra = {"latin_script", "cjk_script", "cyrillic_script", "other_script"}
    return set(probe) | (extra & set(spec.by_name))


def _sentence_spans(text: str) -> list:
    spans, start = [], 0
    for m in v0._SENT_SPLIT.finditer(text):
        if m.start() > start:
            spans.append((start, m.start()))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans or [(0, len(text))]
