"""Weak labelers for R2's text-domain v0 attribute set.

Heuristic, sentence-granular, deliberately coarse — conversion-pipeline.md
§R2: "start coarse and decodable", "label once, reuse forever". Every labeler
maps a sentence to a small class id; positions inherit their sentence's
label. Noisy labels are fine here: R2's probes only need enough signal to
measure decodability, and attributes that stay undecodable are dropped as
findings, not forced.

The core is tokenizer-agnostic: it labels a text reconstructed from per-token
*pieces*, so position alignment is exact by construction. A teacher-LLM
labeler can replace any heuristic later by writing the same
labels/<attr>.npy files (the recording store is the contract).

Classes:
  language        0=latin  1=cjk  2=cyrillic  3=other
  register        0=informal  1=formal
  sentiment       0=negative  1=neutral  2=positive
  topic           0=tech  1=science  2=news/politics  3=other
  speech_act      0=question  1=command  2=statement
  safety_salience 0=benign  1=salient
  entity_presence 0=absent  1=present
"""
from __future__ import annotations

import re

import numpy as np

ATTRIBUTES = ["language", "register", "sentiment", "topic", "speech_act",
              "safety_salience", "entity_presence"]

N_CLASSES = {"language": 4, "register": 2, "sentiment": 3, "topic": 4,
             "speech_act": 3, "safety_salience": 2, "entity_presence": 2}

# ---- lexicons (small, editable; a hit is a vote) ----

_INFORMAL = {"gonna", "wanna", "gotta", "kinda", "yeah", "nope", "ok", "okay",
             "lol", "hey", "stuff", "guys", "cool", "awesome", "btw", "dunno",
             "don't", "can't", "won't", "i'm", "you're", "it's", "that's"}
_FORMAL = {"therefore", "moreover", "furthermore", "consequently", "regarding",
           "pursuant", "hereby", "notwithstanding", "accordingly", "shall",
           "respectively", "aforementioned", "thus", "whereas"}

_POSITIVE = {"good", "great", "excellent", "wonderful", "love", "loved",
             "happy", "beautiful", "best", "amazing", "fantastic", "success",
             "successful", "delighted", "pleased", "enjoy", "enjoyed", "win"}
_NEGATIVE = {"bad", "terrible", "awful", "hate", "hated", "sad", "worst",
             "horrible", "failure", "failed", "angry", "disappointing",
             "disappointed", "broken", "wrong", "problem", "problems", "lose"}

_TECH = {"software", "server", "code", "api", "database", "compiler", "linux",
         "network", "algorithm", "cpu", "gpu", "browser", "app", "computer",
         "programming", "kernel", "cloud", "encryption", "robot", "model"}
_SCIENCE = {"experiment", "hypothesis", "molecule", "quantum", "biology",
            "physics", "chemistry", "neuron", "genome", "theorem", "orbit",
            "species", "cell", "protein", "climate", "galaxy", "enzyme"}
_NEWS = {"government", "election", "president", "minister", "parliament",
         "policy", "senate", "court", "economy", "war", "treaty", "vote",
         "campaign", "congress", "law", "tax", "border"}

_THREAT = {"kill", "killed", "attack", "attacked", "weapon", "weapons", "bomb",
           "gun", "knife", "threat", "danger", "dangerous", "poison", "die",
           "died", "dead", "violence", "violent", "explode", "murder",
           "snake", "fire", "crash", "emergency", "wound", "bleed"}

_COMMAND_VERBS = {"run", "stop", "go", "take", "make", "add", "remove", "open",
                  "close", "click", "install", "delete", "write", "read",
                  "consider", "note", "remember", "check", "use", "try",
                  "please", "let", "set", "put", "turn", "keep", "call"}

_WH = {"what", "who", "where", "when", "why", "how", "which", "whose",
       "is", "are", "do", "does", "did", "can", "could", "would", "will",
       "should", "may", "might"}

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD = re.compile(r"[A-Za-z']+")


def _words(s: str) -> list:
    return [w.lower() for w in _WORD.findall(s)]


def _char_class(s: str) -> int:
    cjk = cyr = latin = other = 0
    for ch in s:
        o = ord(ch)
        if 0x4E00 <= o <= 0x9FFF or 0x3040 <= o <= 0x30FF or 0xAC00 <= o <= 0xD7AF:
            cjk += 1
        elif 0x0400 <= o <= 0x04FF:
            cyr += 1
        elif ch.isascii() and ch.isalpha():
            latin += 1
        elif o > 0x024F and ch.isalpha():
            other += 1
    top = max(latin, cjk, cyr, other)
    if top == 0 or top == latin:
        return 0
    if top == cjk:
        return 1
    if top == cyr:
        return 2
    return 3


def label_sentence(s: str) -> dict:
    """Label one sentence for every attribute."""
    words = _words(s)
    wset = set(words)
    stripped = s.strip()

    # language
    language = _char_class(s)

    # register
    informal = len(wset & _INFORMAL) + s.count("'")
    formal = len(wset & _FORMAL) + sum(1 for w in words if len(w) >= 10)
    register = 1 if formal > informal else 0

    # sentiment
    pos, neg = len(wset & _POSITIVE), len(wset & _NEGATIVE)
    sentiment = 2 if pos > neg else (0 if neg > pos else 1)

    # topic
    votes = [len(wset & _TECH), len(wset & _SCIENCE), len(wset & _NEWS)]
    topic = int(np.argmax(votes)) if max(votes) > 0 else 3

    # speech act
    first = words[0] if words else ""
    if stripped.endswith("?") or (first in _WH and "?" in s):
        speech_act = 0
    elif first in _COMMAND_VERBS or (stripped.endswith("!") and first in _COMMAND_VERBS):
        speech_act = 1
    else:
        speech_act = 2

    # safety salience
    safety_salience = 1 if wset & _THREAT else 0

    # entity presence: capitalized words mid-sentence, or number-heavy tokens
    ents = 0
    toks = stripped.split()
    for i, t in enumerate(toks):
        core = t.strip(".,;:!?\"'()")
        if i > 0 and core[:1].isupper() and core[1:2].islower():
            ents += 1
    entity_presence = 1 if ents > 0 else 0

    return {"language": language, "register": register, "sentiment": sentiment,
            "topic": topic, "speech_act": speech_act,
            "safety_salience": safety_salience,
            "entity_presence": entity_presence}


def label_pieces(pieces: list, attributes: list) -> dict:
    """Label per token position, given per-token text pieces whose
    concatenation is the text (exact alignment by construction). Returns
    {attribute: int64 array [len(pieces)]}."""
    text = "".join(pieces)

    # sentence spans in char space
    spans = []
    start = 0
    for m in _SENT_SPLIT.finditer(text):
        if m.start() > start:
            spans.append((start, m.start()))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    if not spans:
        spans = [(0, len(text))]

    labels_by_span = [label_sentence(text[a:b]) for a, b in spans]

    out = {attr: np.zeros(len(pieces), dtype=np.int64) for attr in attributes}
    # walk tokens through char space, assigning the covering sentence's labels
    ci = 0
    si = 0
    for ti, piece in enumerate(pieces):
        mid = ci + max(1, len(piece)) // 2
        while si + 1 < len(spans) and mid >= spans[si][1]:
            si += 1
        for attr in attributes:
            out[attr][ti] = labels_by_span[si].get(attr, 0)
        ci += len(piece)
    return out


def label_token_windows(tokenizer, windows: list, attributes: list) -> dict:
    """HF glue: decode each token id individually so pieces concatenate to the
    window text, label per window, concatenate across windows."""
    for attr in attributes:
        if attr not in ATTRIBUTES:
            raise SystemExit(f"labelers: unknown attribute '{attr}' (known: {ATTRIBUTES})")
    parts: dict = {attr: [] for attr in attributes}
    for ids in windows:
        pieces = [tokenizer.decode([tid]) for tid in ids]
        got = label_pieces(pieces, attributes)
        for attr in attributes:
            parts[attr].append(got[attr])
    return {attr: np.concatenate(parts[attr]) for attr in attributes}
