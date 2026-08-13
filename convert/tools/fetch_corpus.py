#!/usr/bin/env python3
"""Stream the extraction-v1 seven-stratum corpus from Hugging Face (§2.1) and
write the manifest the stratified loader consumes (§2.2).

Per-stratum files land in corpus/<domain>.txt; corpus/manifest.yaml carries
shares + provenance (dataset, license, fetch date). The behavioral suites
stay with tools/make_corpus.py — run both.

Usage:
  pip install 'datasets>=2.19' pyyaml
  python3 tools/fetch_corpus.py corpus 20000      # ~20 GB total (disk)
  python3 tools/fetch_corpus.py corpus 300        # small bootstrap cut
  python3 tools/fetch_corpus.py corpus 20000 --only=code,math   # refresh strata

Network-heavy: on Modal, run via tools/modal_record.py::fetch_corpus.
"""
import datetime
import os
import sys

import yaml
from datasets import load_dataset

OUT = sys.argv[1] if len(sys.argv) > 1 else "corpus"
TARGET_MB = int(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("-") else 300
ONLY = None
for a in sys.argv[3:]:
    if a.startswith("--only="):
        ONLY = set(a.split("=", 1)[1].split(","))
os.makedirs(OUT, exist_ok=True)

# stratum → (share, [(repo, config, field-candidates)...], license)
# multi-source strata split their byte budget evenly across sources.
FW2_LANGS = ["fra_Latn", "deu_Latn", "spa_Latn", "rus_Cyrl", "ukr_Cyrl",
             "jpn_Jpan", "cmn_Hani", "kor_Hang", "arb_Arab", "hin_Deva"]
STACK_LANGS = ["python", "cpp", "javascript", "rust", "java", "shell"]

STRATA = {
    "web-en":       (0.25, [("HuggingFaceFW/fineweb", "sample-10BT", ["text"])], "odc-by"),
    "multilingual": (0.20, [("HuggingFaceFW/fineweb-2", lang, ["text"]) for lang in FW2_LANGS], "odc-by"),
    "code":         (0.20, [("HuggingFaceTB/stack-edu", lang, ["text", "content"]) for lang in STACK_LANGS], "odc-by"),
    "math":         (0.10, [("HuggingFaceTB/finemath", "finemath-4plus", ["text"])], "odc-by"),
    "science":      (0.10, [("allenai/peS2o", None, ["text"])], "odc-by"),
    "literature":   (0.10, [("deepmind/pg19", None, ["text"])], "apache-2.0"),
    "pdf":          (0.05, [("HuggingFaceFW/finepdfs", "eng_Latn", ["text"])], "odc-by"),
}

target_bytes = TARGET_MB * 1024 * 1024
written_total = 0
strata_entries, provenance = [], []

for domain, (share, sources, license_) in STRATA.items():
    path = os.path.join(OUT, f"{domain}.txt")
    if ONLY and domain not in ONLY:
        if os.path.exists(path):
            strata_entries.append({"domain": domain, "file": f"{OUT}/{domain}.txt", "share": share})
        continue
    budget = int(target_bytes * share)
    per_source = budget // len(sources)
    got_domain = 0
    with open(path, "w", encoding="utf-8") as f:
        for repo, cfg_name, fields in sources:
            print(f"[{domain}] streaming {repo}:{cfg_name or 'default'} "
                  f"(~{per_source / 1024 / 1024:.0f} MB)...", flush=True)
            try:
                ds = load_dataset(repo, name=cfg_name, split="train", streaming=True)
            except Exception as e:  # config drift across dataset versions
                print(f"  skip {repo}:{cfg_name} ({e})")
                continue
            got = 0
            for row in ds:
                doc = ""
                for fld in fields:
                    doc = (row.get(fld) or "").strip()
                    if doc:
                        break
                if len(doc) < 200:
                    continue
                f.write(doc + "\n\n")
                n = len(doc.encode("utf-8")) + 2
                got += n
                if got >= per_source:
                    break
            got_domain += got
            print(f"  wrote {got / 1024 / 1024:.0f} MB from {repo}:{cfg_name or 'default'}")
    written_total += got_domain
    strata_entries.append({"domain": domain, "file": f"{OUT}/{domain}.txt", "share": share})
    for repo, cfg_name, _ in sources:
        provenance.append({"domain": domain, "dataset": repo,
                           **({"config": cfg_name} if cfg_name else {}),
                           "license": license_})

# behavioral suites ride along as their own (tiny) stratum when present
suites = os.path.join(OUT, "behavioral-suites.txt")
if os.path.exists(suites):
    strata_entries.append({"domain": "behavioral", "file": f"{OUT}/behavioral-suites.txt",
                           "share": 0.001})
    provenance.append({"domain": "behavioral",
                       "dataset": "tools/make_corpus.py (synthetic)",
                       "license": "project"})

manifest = {"strata": strata_entries, "provenance": provenance,
            "fetched": datetime.date.today().isoformat(),
            "target_mb": TARGET_MB}
with open(os.path.join(OUT, "manifest.yaml"), "w") as f:
    yaml.safe_dump(manifest, f, sort_keys=False)

print(f"corpus: {written_total / 1024 / 1024:.0f} MB across "
      f"{len(strata_entries)} strata → {OUT}/manifest.yaml")
ok = ONLY is not None or written_total >= 0.4 * target_bytes
print("FETCH_CORPUS: OK" if ok else
      f"FETCH_CORPUS: FAILED ({written_total / 1024 / 1024:.0f} MB of {TARGET_MB} MB target)")
sys.stdout.flush()
os._exit(0 if ok else 1)   # datasets' streaming threads stall normal exit
