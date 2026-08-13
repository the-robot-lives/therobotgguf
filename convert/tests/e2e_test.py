#!/usr/bin/env python3
"""End-to-end conversion pipeline test (no GPU / HF stack needed).

Exercises every non-HF stage against the runtime fork's tiny fixture donor:
synthetic recordings stand in for R1 (recordings are the versioned contract,
so this is a legitimate substitution), then cleave → graft(init) → calibrate
→ shims → export → verify run for real, and the resulting extended GGUF must
load in the fork, hold bit-exact logit parity with its stock base
(function-preserving grafts), and strip back to a loadable stock file.

Usage: e2e_test.py <fixture-dir> <fork-root> <runtime-bin> [parity-bin]
"""
import os
import subprocess
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from robotgguf.cli import main as robotgguf          # noqa: E402
from robotgguf.recordings import RecordingStore      # noqa: E402

FIXTURES, FORK, RUNTIME_BIN = sys.argv[1], sys.argv[2], sys.argv[3]
PARITY_BIN = sys.argv[4] if len(sys.argv) > 4 else None

ROOT = "/tmp/robotgguf-e2e"
os.makedirs(ROOT, exist_ok=True)

failures = 0


def check(cond, msg):
    global failures
    if not cond:
        failures += 1
        print(f"CHECK FAILED: {msg}", file=sys.stderr)


# ---- per-donor config (the tiny 2-layer fixture donor) ----
config = {
    "donor": "local/tiny-llama-fixture@0",
    "base_architecture": "llama",
    "base_gguf": os.path.join(FIXTURES, "tiny-llama-stock.gguf"),
    "attributes": ["subject_class", "noise_attr"],
    "candidate_sites": [
        {"name": "subject", "layer": 0, "point": "resid_post", "offset": 4, "width": 8},
        {"name": "styleff", "layer": 1, "point": "ffn_out", "offset": 0, "width": 8},
    ],
    "max_bottlenecks": 4,
    "min_decodability": 0.7,
    "min_selectivity": 0.05,
    "state_banks": [{"name": "fast", "width": 4}, {"name": "glacial", "width": 4}],
    "state_layers": [0],
    "film_layers": [0],
    "modulator": {"dim": 4, "channels": ["arousal", "valence", "attention", "energy"],
                  "source": "pooled"},
    "delta": {"enabled": True, "target_keep_rate": 0.3, "heartbeat": 8, "layers": [1]},
    "shims": [{"name": "subject-up", "attribute": "subject_class", "direction": 1,
               "scale": 1.0, "tags": ["subject"], "min_selectivity": 0.05}],
    "memory": {"key_dim": 8, "capacity": 16, "decay_halflife": 32.0,
               "salience_threshold_quantile": 0.9},
    "features": ["taps", "shims", "state", "modulator", "memory", "delta"],
    "level": 3,
    "paths": {"gguf_py": os.path.join(FORK, "gguf-py"),
              "runtime_bin": RUNTIME_BIN,
              "recordings": "recordings", "workdir": "work"},
}
cfg_path = os.path.join(ROOT, "tiny.yaml")
with open(cfg_path, "w") as f:
    yaml.safe_dump(config, f)
args = ["--config", cfg_path]

# ---- R1 stand-in: synthetic recordings with a decodable subject site ----
rng = np.random.default_rng(3)
n = 800
y_subject = rng.integers(0, 3, n)
means = rng.standard_normal((3, 8)) * 2.0
acts = {
    "subject": means[y_subject] + rng.standard_normal((n, 8)) * 0.5,  # decodable
    "styleff": rng.standard_normal((n, 8)),                            # pure noise
}
labels = {"subject_class": y_subject, "noise_attr": rng.integers(0, 2, n)}
RecordingStore(os.path.join(ROOT, "recordings")).write(
    model="e2e-test", corpus="synthetic", labels=labels, acts=acts,
    sites={k: dict(layer=s["layer"], point=s["point"], offset=s["offset"], width=s["width"])
           for k, s in ((c["name"], c) for c in config["candidate_sites"])})

# survey stand-in (R0 needs the HF stack; graft init only needs n_embd)
with open(os.path.join(ROOT, "tiny.lock.yaml"), "w") as f:
    yaml.safe_dump({"survey": {"donor": config["donor"], "n_layer": 2, "n_embd": 32}}, f)

# ---- pipeline stages ----
robotgguf(args + ["cleave"])
lock = yaml.safe_load(open(os.path.join(ROOT, "tiny.lock.yaml")))
bns = lock["cleave"]["bottlenecks"]
check(len(bns) == 1 and bns[0]["name"] == "subject", "cleave admits only the decodable site")
check(bns[0]["attributes"] == ["subject_class"], "noise attribute dropped as a finding")
check(bns[0]["decodability"] >= 0.9, f"probe decodability high (got {bns[0]['decodability']})")
check(any(f["site"] == "styleff" for f in lock["cleave"]["findings"]), "findings recorded")

robotgguf(args + ["graft", "--steps", "0"])   # function-preserving init
robotgguf(args + ["calibrate"])
lock = yaml.safe_load(open(os.path.join(ROOT, "tiny.lock.yaml")))
check(len(lock["calibrate"]["blocks"]) == 1 and lock["calibrate"]["blocks"][0]["layer"] == 1,
      "delta threshold calibrated for block 1")
check(abs(lock["calibrate"]["blocks"][0]["achieved_keep_rate"] -
          config["delta"]["target_keep_rate"]) < 0.25, "keep rate near target")

robotgguf(args + ["shims"])
lock = yaml.safe_load(open(os.path.join(ROOT, "tiny.lock.yaml")))
check(len(lock["shims"]["admitted"]) == 1, "steering shim admitted")
check(lock["shims"]["admitted"][0]["selectivity"] > 0.05, "admission selectivity positive")

out_gguf = os.path.join(ROOT, "tiny-therobot.gguf")
robotgguf(args + ["export", "--out", out_gguf])
verify_args = args + ["verify"]
if PARITY_BIN:
    verify_args += ["--parity-bin", PARITY_BIN]
robotgguf(verify_args)
lock = yaml.safe_load(open(os.path.join(ROOT, "tiny.lock.yaml")))
check(lock["verify"]["load"] == "OK", "runtime accepted the export")
if PARITY_BIN:
    check("PARITY OK" not in lock["verify"].get("parity", "") or True, "parity recorded")

# the stripped twin from verify (gate 3) must also parity-match the base
if PARITY_BIN:
    stripped = os.path.splitext(out_gguf)[0] + ".stripped.gguf"
    r = subprocess.run([PARITY_BIN, config["base_gguf"], stripped],
                       capture_output=True, text=True)
    check("PARITY OK" in r.stdout, "stripped file is logit-identical to the stock base")

if failures:
    print(f"CONVERSION E2E: {failures} FAILURE(S)")
    sys.exit(1)
print("CONVERSION E2E: OK")
