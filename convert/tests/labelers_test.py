#!/usr/bin/env python3
"""Weak-labeler unit tests — sentence classification sanity + exact per-token
alignment. No HF stack needed (the core is tokenizer-agnostic)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from robotgguf.labelers import ATTRIBUTES, N_CLASSES, label_pieces, label_sentence  # noqa: E402

failures = 0


def check(cond, msg):
    global failures
    if not cond:
        failures += 1
        print(f"CHECK FAILED: {msg}", file=sys.stderr)


# ---- sentence-level classifications ----
cases = [
    # (sentence, attribute, expected)
    ("What time does the server restart?", "speech_act", 0),
    ("Install the compiler and run the tests.", "speech_act", 1),
    ("The weather was mild yesterday.", "speech_act", 2),
    ("The new API and database code runs on the gpu cluster.", "topic", 0),
    ("The experiment confirmed the hypothesis about the protein.", "topic", 1),
    ("The president addressed parliament about the election.", "topic", 2),
    ("We had lunch at noon.", "topic", 3),
    ("This is an excellent, wonderful result and I love it.", "sentiment", 2),
    ("It was a terrible, horrible failure.", "sentiment", 0),
    ("The box contains twelve items.", "sentiment", 1),
    ("Notwithstanding the aforementioned considerations, the committee shall proceed accordingly.", "register", 1),
    ("yeah that's kinda cool lol, gonna check it later", "register", 0),
    ("He saw a snake and grabbed a knife in a dangerous panic.", "safety_salience", 1),
    ("She planted tulips along the fence.", "safety_salience", 0),
    ("Yesterday Alice met Bob in Paris.", "entity_presence", 1),
    ("the quick brown fox jumps over the lazy dog", "entity_presence", 0),
    ("The quick brown fox jumps over the lazy dog.", "language", 0),
    ("今日は天気がいいですね、散歩に行きましょう。", "language", 1),
    ("Сегодня хорошая погода для прогулки.", "language", 2),
]
for sent, attr, expect in cases:
    got = label_sentence(sent)[attr]
    check(got == expect, f"{attr}({sent!r}) = {got}, expected {expect}")

# every attribute stays inside its declared class range
for sent, _, _ in cases:
    lab = label_sentence(sent)
    for attr in ATTRIBUTES:
        check(0 <= lab[attr] < N_CLASSES[attr], f"{attr} class {lab[attr]} out of range")

# ---- per-token alignment across a sentence boundary ----
pieces = ["What", " is", " this", "?", " Install", " the", " update", "."]
labels = label_pieces(pieces, ["speech_act"])["speech_act"]
check(labels.shape == (8,), "one label per piece")
check(all(labels[:4] == 0), f"question sentence tokens labeled 0 (got {labels[:4]})")
check(all(labels[4:] == 1), f"command sentence tokens labeled 1 (got {labels[4:]})")

# mixed attributes at once, deterministic
out = label_pieces(pieces, ATTRIBUTES)
check(sorted(out) == sorted(ATTRIBUTES), "all attributes labeled")
out2 = label_pieces(pieces, ATTRIBUTES)
check(all(np.array_equal(out[a], out2[a]) for a in ATTRIBUTES), "deterministic")

# degenerate inputs don't crash
label_pieces([""], ATTRIBUTES)
label_pieces(["   "], ATTRIBUTES)

if failures:
    print(f"LABELERS TEST: {failures} FAILURE(S)")
    sys.exit(1)
print("LABELERS TEST: OK")
