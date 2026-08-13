#!/usr/bin/env python3
"""Build a mixed calibration corpus deliberately balanced across the seven v0
attributes (conversion-pipeline.md §R2). Balanced synthetic text is a better
R2 substrate than random web scrape: it guarantees every attribute has class
variation, so cleave can actually measure decodability instead of drowning in
a single dominant class. Real prose (a Wikipedia sample) is mixed in for
naturalness.

Writes corpus/mixed-text.txt and corpus/behavioral-suites.txt.
"""
import os
import random
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "corpus"
os.makedirs(OUT, exist_ok=True)
rng = random.Random(20260708)

# ---- building blocks spanning the attribute axes ----

TECH_S = [
    "The server restarted after the kernel panic and the database reconnected.",
    "We refactored the API so the compiler inlines the hot path on the gpu.",
    "The browser cached the response and the algorithm converged in linear time.",
    "Our software pipeline builds the docker image and pushes it to the cluster.",
    "The encryption library rotated its keys and the network latency dropped.",
]
SCI_S = [
    "The experiment confirmed the hypothesis about the enzyme's binding site.",
    "Each neuron fired when the molecule crossed the cell membrane.",
    "The quantum state decohered before the physics apparatus could measure it.",
    "The genome sequence revealed a mutation in the protein-coding region.",
    "Astronomers observed the galaxy's orbit shift under gravitational lensing.",
]
NEWS_S = [
    "The president addressed parliament about the election and the new tax policy.",
    "The senate passed the treaty after a long campaign and a close vote.",
    "The court ruled that the government's border policy violated the law.",
    "The minister resigned as the economy entered its third quarter of decline.",
    "Voters gathered at the capital to protest the proposed congressional budget.",
]
OTHER_S = [
    "She planted tulips along the garden fence before lunch.",
    "The old cafe on the corner still serves the best coffee in town.",
    "We walked along the river and watched the boats drift past.",
    "The children built a sandcastle while the tide slowly came in.",
    "He folded the laundry and set the kettle on to boil.",
]

POS = ["This is an excellent and wonderful result that I truly love.",
       "The team delivered a fantastic, successful launch and everyone was delighted.",
       "What a beautiful, amazing morning; the best I can remember."]
NEG = ["This was a terrible, awful failure and I hate how it turned out.",
       "The broken build was a horrible disappointment for the whole team.",
       "It was the worst, saddest outcome we could have imagined."]

FORMAL = [
    "Notwithstanding the aforementioned considerations, the committee shall proceed accordingly.",
    "Furthermore, the parties hereby agree that the obligations remain in effect respectively.",
    "Pursuant to the regulation, the applicant must therefore submit the documentation forthwith.",
]
INFORMAL = [
    "yeah that's kinda cool, gonna check it out later lol.",
    "hey guys, dunno if it's awesome but it's ok i guess.",
    "nope, that stuff won't work, we'll just wing it btw.",
]

QUESTION = ["What time does the server restart tonight?",
            "Who approved the new border policy and why?",
            "How does the enzyme fold under high temperature?"]
COMMAND = ["Install the compiler and run the full test suite.",
           "Please consider the safety implications before you deploy.",
           "Remember to rotate the keys and check the logs."]

THREAT = [
    "He saw a snake in the leaves and grabbed a knife in a dangerous panic.",
    "The emergency alarm blared as the fire spread and people ran from the crash.",
    "The attacker raised the weapon and the crowd feared a violent, deadly threat.",
]
BENIGN = [
    "The librarian sorted the returned books onto the wooden shelves.",
    "A gentle breeze moved the curtains as the afternoon light warmed the room.",
    "They shared a quiet meal and talked about the weekend hike.",
]

ENTITY = ["Yesterday Alice met Bob near the Eiffel Tower in Paris.",
          "President Lincoln spoke in Gettysburg during November.",
          "Microsoft and Google announced a deal in Seattle on Tuesday."]
NOENTITY = ["the quick brown fox jumps over the lazy dog again and again.",
            "a small bird sang in the tall green tree by the road.",
            "several old boxes sat in the dusty corner of the room."]

# multilingual (language attribute)
JA = ["今日は天気がいいですね、散歩に行きましょう。",
      "この本はとても面白いと思います。",
      "駅までの道を教えてくれますか。"]
RU = ["Сегодня хорошая погода для долгой прогулки.",
      "Эта книга очень интересная и полезная.",
      "Скажите, пожалуйста, где находится вокзал."]

# real prose sample (fetched from simple.wikipedia.org/wiki/Democracy)
REAL = (
    "Democracy means rule by the people. There are different ways this can be done. "
    "People meet to decide about new laws, and changes to existing ones. This is usually "
    "called direct democracy. It is never used except in small countries, or perhaps in "
    "towns. Modern populations are usually too large to do this. The people elect their "
    "leaders. These leaders make decisions about laws. This is called representative "
    "democracy. Elections are either held after a certain time, or when a leader dies. "
    "The type of government where only one person has most of the power is called a "
    "dictatorship. Democracy is the opposite of a dictatorship. Democracy was developed "
    "long ago by the Greeks in classical Athens. There, everyone who was a citizen got "
    "together in one area. The assembly would talk about what kinds of laws they wanted "
    "and voted on them. In the Middle Ages, there were many systems, although only a few "
    "people could join in at this time. The Parliament of England began with Magna Carta, "
    "a document which said that the King's power was limited, and protected certain rights "
    "of the people."
)

ALL = (TECH_S + SCI_S + NEWS_S + OTHER_S + POS + NEG + FORMAL + INFORMAL +
       QUESTION + COMMAND + THREAT + BENIGN + ENTITY + NOENTITY + JA + RU)

# ---- synth-mixed.txt: shuffled balanced paragraphs (a SEPARATE file so it
# never clobbers a fetched corpus/mixed-text.txt). Only written when there is
# no real mixed-text.txt, or when --force is given. ----
mixed_path = os.path.join(OUT, "mixed-text.txt")
synth_path = os.path.join(OUT, "synth-mixed.txt")
force = "--force" in sys.argv
target_chars = int(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("-") else 400_000

paras = []
while sum(len(p) for p in paras) < target_chars:
    block = rng.sample(ALL, k=rng.randint(3, 6))
    if rng.random() < 0.15:
        block.append(REAL)
    paras.append(" ".join(block))

if os.path.exists(mixed_path) and os.path.getsize(mixed_path) > 2 * target_chars and not force:
    # a real (fetched) corpus is present — write the synthetic beside it so
    # both can be listed under `corpus:`; do not touch the fetched file
    with open(synth_path, "w") as f:
        f.write("\n\n".join(paras) + "\n")
    print(f"corpus: kept existing mixed-text.txt; wrote balanced synth-mixed.txt "
          f"({os.path.getsize(synth_path)/1024:.0f} KB) — add it to the config's corpus: list")
else:
    with open(mixed_path, "w") as f:
        f.write("\n\n".join(paras) + "\n")

# ---- behavioral-suites.txt: dense blocks that stress specific attributes,
# so the priming/temporal probes later have material (planning M3 flavor) ----
suites = []
suites.append("\n".join(QUESTION * 6))
suites.append("\n".join(COMMAND * 6))
suites.append("\n".join(THREAT * 6))          # the "snake in the hose" salience block
suites.append("\n".join(FORMAL * 6))
suites.append("\n".join(INFORMAL * 6))
suites.append("\n".join((POS + NEG) * 4))
with open(os.path.join(OUT, "behavioral-suites.txt"), "w") as f:
    f.write("\n\n".join(suites) + "\n")

mt = os.path.getsize(mixed_path)
bt = os.path.getsize(os.path.join(OUT, "behavioral-suites.txt"))
print(f"corpus: mixed-text.txt {mt/1024:.0f} KB, behavioral-suites.txt {bt/1024:.0f} KB → {OUT}/")

# explicit verdict: behavioral suites must be non-trivial, and mixed-text must
# exist (fetched or synthetic) with real content
ok = bt > 1024 and mt > 1024
if os.path.exists(synth_path):
    print(f"        synth-mixed.txt {os.path.getsize(synth_path)/1024:.0f} KB "
          f"(optional — add to the config's corpus: list to layer it on)")
if ok:
    print("MAKE_CORPUS: OK")
    sys.exit(0)
print("MAKE_CORPUS: FAILED (empty output)", file=sys.stderr)
sys.exit(1)
