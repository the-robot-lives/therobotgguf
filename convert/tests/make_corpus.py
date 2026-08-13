#!/usr/bin/env python3
"""Assemble a mixed calibration corpus with coverage across the v0 attribute
axes (register, sentiment, topic, speech-act, safety salience, entities,
language). Sources: real technical prose harvested from installed Python
docstrings and system docs, Faker-generated everyday/multilingual text, and
project-authored blocks for the axes that need deliberate variation. Also
writes the behavioral-suites file (always project-authored).

Usage: make_corpus.py <out-dir> [target_kb]
"""
import inspect
import io
import os
import random
import re
import sys

random.seed(7)
OUT = sys.argv[1]
TARGET_KB = int(sys.argv[2]) if len(sys.argv) > 2 else 600
os.makedirs(OUT, exist_ok=True)

parts = []


def add(tag, text):
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    if len(text) > 200:
        parts.append(text)


# ---- 1. real technical prose: docstrings from installed libraries ----
def harvest_docstrings(modnames, budget):
    got = 0
    for name in modnames:
        try:
            mod = __import__(name)
        except Exception:
            continue
        for _, obj in inspect.getmembers(mod):
            doc = inspect.getdoc(obj)
            if doc and len(doc) > 300:
                add("tech", doc)
                got += len(doc)
                if got > budget:
                    return


harvest_docstrings(["json", "os", "re", "subprocess", "logging", "argparse",
                    "difflib", "statistics", "email", "http", "socket",
                    "threading", "unittest", "pickle", "csv", "sqlite3"], 80_000)
harvest_docstrings(["numpy", "numpy.linalg", "numpy.fft", "numpy.random"], 80_000)

# ---- 2. real English from system docs ----
for root in ("/usr/share/doc", "/usr/share/common-licenses"):
    budget = 60_000
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if budget <= 0:
                break
            if fn.lower().startswith(("readme", "copyright", "gpl", "apache")):
                try:
                    with io.open(os.path.join(dirpath, fn), errors="ignore") as f:
                        t = f.read(8000)
                    if t.count(" ") > 100:
                        add("doc", t)
                        budget -= len(t)
                except OSError:
                    pass

# ---- 3. Faker: everyday text, entities, RU/JA ----
from faker import Faker  # noqa: E402

for locale, n in (("en_US", 300), ("ru_RU", 60), ("ja_JP", 60)):
    fk = Faker(locale)
    fk.seed_instance(11)
    buf = []
    for _ in range(n):
        who = f"{fk.name()} of {fk.company()}" if locale == "en_US" else fk.name()
        buf.append(f"{who} said: {fk.paragraph(nb_sentences=4)}")
    add("faker-" + locale, "\n".join(buf))

# ---- 4. project-authored variation blocks ----
NEWS = """The government announced a new policy on energy taxation before the
election. Parliament debated the treaty for three days while the president met
with the ministers. The senate vote on the border law was postponed after the
court ruling, and the campaign shifted its economic message. Congress passed
the budget despite objections; the economy grew slowly through the quarter.
Analysts said the election would hinge on tax policy and the new trade treaty.
The minister resigned after the vote, and the parliament scheduled hearings."""

SCIENCE = """The experiment measured how the protein folds when the enzyme
concentration changes. Under the microscope, each cell showed the same pattern
the hypothesis predicted. Quantum effects dominate at that scale, and the
molecule's orbit around the binding site follows the theorem's bound. The
genome survey identified a species whose neurons regenerate; climate records
from the glacier cores confirmed the physics model. Chemistry constrains what
the biology can do, and the galaxy survey gave the astronomers new data."""

INFORMAL = """yeah so i'm gonna grab lunch, wanna come? that new place is kinda
awesome tbh. lol ok but don't be late again