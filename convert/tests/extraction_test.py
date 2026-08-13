#!/usr/bin/env python3
"""extraction-v1 unit tests (C1 corpus, C2 semvec/labelers, C4 cleave-vec).

No HF stack needed: a stub tokenizer stands in for R1's, and cleave-vec runs
on synthetic recordings (recordings are the versioned contract, same policy
as tests/e2e_test.py).

  python3 tests/extraction_test.py /tmp/robot-extraction-test
  # expect: EXTRACTION TESTS: OK
"""
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from robotgguf import corpus as corpus_mod  # noqa: E402
from robotgguf import cleave_vec, labelers_t0, semvec  # noqa: E402
from robotgguf.config import Config  # noqa: E402
from robotgguf.recordings import RecordingStore  # noqa: E402

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/robot-extraction-test"
SPEC_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "semvec-v1.yaml")
shutil.rmtree(ROOT, ignore_errors=True)
os.makedirs(ROOT)
FAILURES = []


def check(name, cond, detail=""):
    print(f"  {'ok' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


class StubTok:
    """Whitespace 'tokenizer' with the HF call/decode surface the loader and
    labelers use. Ids index a growing vocab; decode returns the piece + a
    trailing space (concatenation reconstructs the text closely enough)."""

    def __init__(self):
        self.vocab, self.inv = {}, []

    def __call__(self, text):
        ids = []
        for w in text.split():
            if w not in self.vocab:
                self.vocab[w] = len(self.inv)
                self.inv.append(w)
            ids.append(self.vocab[w])
        return type("Enc", (), {"input_ids": ids})()

    def decode(self, ids):
        return " ".join(self.inv[i] for i in ids) + " "


class StubCfg:
    def __init__(self, corpus):
        self.corpus = corpus

    def resolve(self, p):
        return p if os.path.isabs(p) else os.path.abspath(p)


# ---------------------------------------------------------------- C1: corpus
print("C1 — manifest + stratified loader")
cdir = os.path.join(ROOT, "corpus")
os.makedirs(cdir)
words = {"alpha": "aa bb cc dd ee ff gg hh", "beta": "ii jj kk ll mm nn oo pp",
         "gamma": "qq rr ss tt uu vv ww xx"}
for dom, ws in words.items():
    with open(os.path.join(cdir, f"{dom}.txt"), "w") as f:
        f.write((ws + " ") * 3000)          # plenty of tokens
man_path = os.path.join(cdir, "manifest.yaml")
with open(man_path, "w") as f:
    import yaml
    yaml.safe_dump({"strata": [
        {"domain": "alpha", "file": os.path.join(cdir, "alpha.txt"), "share": 0.5},
        {"domain": "beta", "file": os.path.join(cdir, "beta.txt"), "share": 0.3},
        {"domain": "gamma", "file": os.path.join(cdir, "gamma.txt"), "share": 0.2},
    ]}, f)

strata = corpus_mod.load_strata(StubCfg([man_path]))
check("manifest loads 3 strata", len(strata) == 3)
check("shares normalized", abs(sum(s.share for s in strata) - 1.0) < 1e-9)

tok = StubTok()
wins, dom_ids, doms = corpus_mod.stratified_windows(strata, tok, max_tokens=10_000, window=50)
shares = corpus_mod.domain_shares_report(dom_ids, doms)
check("window count", len(wins) == 200, str(len(wins)))
check("all windows full", all(len(w) == 50 for w in wins))
check("share alpha ±2%", abs(shares["alpha"] - 0.5) <= 0.02, str(shares))
check("share beta ±2%", abs(shares["beta"] - 0.3) <= 0.02, str(shares))
check("share gamma ±2%", abs(shares["gamma"] - 0.2) <= 0.02, str(shares))

wins2, dom2, _ = corpus_mod.stratified_windows(strata, StubTok(), max_tokens=10_000, window=50)
check("deterministic (same seed)", wins == wins2 and (dom_ids == dom2).all())

# exhaustion: tiny stratum renormalizes instead of starving the budget
with open(os.path.join(cdir, "tiny.txt"), "w") as f:
    f.write("zz " * 120)                    # ~2 windows worth
strata_x = strata + [corpus_mod.Stratum("tiny", os.path.join(cdir, "tiny.txt"), 0.5)]
total = sum(s.share for s in strata_x)
for s in strata_x:
    s.share /= total
wx, dx, domx = corpus_mod.stratified_windows(strata_x, StubTok(), max_tokens=8_000, window=50)
check("exhausted stratum renormalizes", len(wx) == 160,
      f"{len(wx)} windows, shares {corpus_mod.domain_shares_report(dx, domx)}")

# v0 bare-list back-compat
strata_v0 = corpus_mod.load_strata(StubCfg([os.path.join(cdir, "alpha.txt"),
                                            os.path.join(cdir, "beta.txt")]))
check("bare list → equal-share strata", len(strata_v0) == 2 and
      all(abs(s.share - 0.5) < 1e-9 for s in strata_v0))

# ---------------------------------------------------------------- C2: semvec
print("C2 — semvec spec + views + t0")
spec = semvec.SemvecSpec.load(SPEC_PATH)
check("named_dim covered exactly", len(spec.axes) == spec.named_dim)
check("dim = named + latent", spec.dim == spec.named_dim + spec.latent_dim)
h1 = spec.spec_hash()
check("spec hash stable", h1 == semvec.SemvecSpec.load(SPEC_PATH).spec_hash())
check("v0 views all present", all(a in spec.views for a in
      ["language", "register", "sentiment", "topic", "speech_act",
       "safety_salience", "entity_presence"]))

b = semvec.VectorBuilder(spec, 4)
b.set_axis("formality", np.array([0, 1, 3, 4]), tier="t0")
b.set_axis("formality", np.array([9, 9, 9, 9]), tier="t2")   # clipped + overrides
b.set_axis("formality", np.array([1, 1, 1, 1]), tier="t1")   # must NOT clobber t2
v = b.finish()
check("tier precedence t2 > t1 > t0", (v[:, spec.axis("formality").index] == 4.0).all())
check("sources recorded", b.sources["formality"] == "t2")

vv = np.zeros((5, spec.dim), dtype=np.float32)
vv[:, spec.axis("valence").index] = [0.0, 1.4, 2.0, 2.6, 4.0]
vv[:, spec.axis("question_ness").index] = [4, 0, 0, 0, 0]
vv[:, spec.axis("imperative_ness").index] = [0, 4, 0, 0, 0]
views = semvec.categorical_views(spec, vv, ["sentiment", "speech_act"])
check("sentiment thresholds", views["sentiment"].tolist() == [0, 0, 1, 2, 2])
check("speech_act argmax+fallback", views["speech_act"].tolist() == [0, 1, 2, 2, 2])

# t0 axis sanity on real-ish sentences
sc_q = labelers_t0.score_sentence(spec, "What time does the server restart tonight?")
sc_code = labelers_t0.score_sentence(spec, "def main(): return compile(x); // fn")
sc_plain = labelers_t0.score_sentence(spec, "She planted tulips along the garden fence.")
check("t0 question_ness fires", sc_q["question_ness"] == 4.0)
check("t0 code_ness orders", sc_code["code_ness"] > sc_plain["code_ness"])
check("t0 script one-hot", labelers_t0.score_sentence(spec, "今日は天気がいいですね。")["cjk_script"] == 4.0)

# basis freeze + reduce roundtrip + pin check
rng = np.random.default_rng(0)
emb = rng.normal(size=(600, 48)).astype(np.float32)
basis_path = os.path.join(ROOT, "basis.npz")
lat_spec = semvec.SemvecSpec.load(SPEC_PATH)
lat_spec.latent = dict(lat_spec.latent or {})
sha = semvec.fit_basis(emb, latent_dim=16, out_path=basis_path)
lat_spec.latent.update({"basis": basis_path, "basis_sha256": sha})
lat_spec.latent_dim = 16
mean_comp = semvec.load_basis(lat_spec)
red = semvec.reduce_embeddings(emb, mean_comp)
check("reduce shape", red.shape == (600, 16))
check("whitened-ish", abs(float(red.std()) - 1.0) < 0.2, f"std={red.std():.3f}")
lat_spec.latent["basis_sha256"] = "0" * 64
try:
    semvec.load_basis(lat_spec)
    check("basis pin enforced", False)
except SystemExit:
    check("basis pin enforced", True)

# ------------------------------------------------------------ C4: cleave-vec
print("C4 — ridge map + per-axis admission")
rec_dir = os.path.join(ROOT, "recordings")
n, d = 6000, 32
x = rng.normal(size=(n, d)).astype(np.float32)
vec = np.zeros((n, spec.dim), dtype=np.float32)
# driven named axes: formality ← x[:,0], menace ← nonlinear |x[:,1]| (MLP-only)
fi, mi = spec.axis("formality").index, spec.axis("menace").index
vec[:, fi] = np.clip(2.0 + 1.2 * x[:, 0], 0, 4)
vec[:, mi] = np.clip(3.0 * np.abs(x[:, 1]) - 1.0, 0, 4)
# a driven latent axis
vec[:, spec.named_dim + 3] = 0.9 * x[:, 2]
# an undriven-but-varying control axis
vec[:, spec.axis("humor").index] = rng.uniform(0, 4, size=n)
domains = (np.arange(n) // (n // 3)).clip(0, 2).astype(np.int64)

store = RecordingStore(rec_dir)
store.write(model="stub", corpus="stub",
            sites={"s0": {"layer": 2, "point": "resid_post", "offset": 0, "width": d}},
            acts={"s0": x.astype(np.float16)},
            labels={"domain": domains}, domain_names=["a", "b", "c"],
            semvec={"version": spec.version, "hash": spec.spec_hash()})
np.save(os.path.join(rec_dir, "labels", "vector.npy"), vec.astype(np.float16))

cfg_yaml = os.path.join(ROOT, "cfg.yaml")
with open(cfg_yaml, "w") as f:
    import yaml
    yaml.safe_dump({"donor": "stub", "semvec": os.path.abspath(SPEC_PATH),
                    "attributes": [],
                    "vec_sample_cap": 6000, "mlp_fallback_max": 4,
                    "paths": {"recordings": rec_dir,
                              "workdir": os.path.join(ROOT, "work")}}, f)
cfg = Config.load(cfg_yaml)
cleave_vec.run(cfg)

import yaml as _yaml
lock = _yaml.safe_load(open(os.path.join(ROOT, "cfg.lock.yaml")))
cv = lock["cleave_vec"]
admitted = {a["axis"] for a in cv["sites"]["s0"]["axes"]}
check("driven named axis admitted", "formality" in admitted, str(sorted(admitted))[:120])
check("driven latent axis admitted", "latent_3" in admitted)
check("undriven axis rejected", "humor" not in admitted)
check("zero-variance axes rejected", "grief" not in admitted)
nl = [f for f in cv["findings"] if f["axis"] == "menace"]
check("nonlinear axis → finding", len(nl) == 1 and nl[0]["mlp"] > nl[0]["linear"],
      str(nl))
proj = np.load(os.path.join(cv["probe_dir"], "s0.proj.npy"))
calib = np.load(os.path.join(cv["probe_dir"], "s0.calib.npy"))
check("proj shape", proj.shape == (d, spec.dim))
check("unadmitted columns zeroed", float(np.abs(proj[:, spec.axis("humor").index]).max()) == 0.0)
check("admitted calib scale=1", float(calib[fi, 0]) == 1.0)
# readout sanity: proj recovers the driven axis on held-out-ish data
pred = x @ proj[:, fi] + calib[fi, 1]
r = np.corrcoef(pred, vec[:, fi])[0, 1]
check("readout recovers formality", r > 0.9, f"r={r:.3f}")

# overlay: the write path (standardized layer → slice) — extraction-v1 §4.4
from robotgguf import overlay as overlay_mod  # noqa: E402
gmat = np.load(os.path.join(cv["probe_dir"], "s0.overlay.npy"))
check("overlay shape", gmat.shape == (spec.dim, d))
site_ax = {a["axis"]: a for a in cv["sites"]["s0"]["axes"]}
check("admitted axes writable", site_ax["formality"]["writable"] and
      site_ax["latent_3"]["writable"], str(site_ax))
h = x[:64]
s0 = overlay_mod.read(h, proj, calib[:, 1])
zero = np.zeros((64, spec.dim), dtype=np.float32)
check("Δs=0 is identity", np.array_equal(overlay_mod.apply_overlay(h, zero, gmat), h))
ds = zero.copy()
ds[:, fi] = 1.0                       # portable module says: +1 formality
h2 = overlay_mod.apply_overlay(h, ds, gmat)
s1 = overlay_mod.read(h2, proj, calib[:, 1])
gain = float((s1[:, fi] - s0[:, fi]).mean())
check("unit write → unit readout", abs(gain - 1.0) < 1e-3, f"gain={gain:.4f}")
li = spec.named_dim + 3
xt = float(np.abs(s1[:, li] - s0[:, li]).mean())
check("crosstalk onto latent_3 small", xt < 0.2, f"xt={xt:.4f}")
check("unwritable rows zeroed", float(np.abs(gmat[spec.axis("humor").index]).max()) == 0.0)
check("semvec hash mismatch refused", True)  # covered by manifest guard below
store2 = RecordingStore(rec_dir)
m = store2.manifest
m.semvec = {"version": spec.version, "hash": "deadbeefdeadbeef"}
m.save(rec_dir)
try:
    cleave_vec.run(cfg)
    check("hash guard fires", False)
except SystemExit:
    check("hash guard fires", True)

# ------------------------------------------------- R7: semvec export + strip
print("R7 — semvec packaging (needs gguf-py)")
try:
    import gguf  # noqa: F401
    HAVE_GGUF = True
except ImportError:
    HAVE_GGUF = False
    print("  skip  (pip install gguf, or point paths.gguf_py at the fork)")

if HAVE_GGUF:
    import gguf
    from robotgguf import export as export_mod
    from robotgguf import shimc

    # restore the manifest hash the guard test broke
    m = RecordingStore(rec_dir).manifest
    m.semvec = {"version": spec.version, "hash": spec.spec_hash()}
    m.save(rec_dir)

    # tiny stock base file (one dummy tensor + the KV export reads)
    base_path = os.path.join(ROOT, "base.gguf")
    bw = gguf.GGUFWriter(base_path, "llama")
    bw.add_uint32("llama.embedding_length", 64)
    bw.add_tensor("dummy.weight", np.zeros((4, 4), dtype=np.float32))
    bw.write_header_to_file(); bw.write_kv_data_to_file(); bw.write_tensors_to_file(); bw.close()

    with open(cfg_yaml) as f:
        import yaml
        raw = yaml.safe_load(f)
    raw.update({"base_architecture": "llama", "base_gguf": base_path,
                "features": [], "candidate_sites": [
                    {"name": "s0", "layer": 2, "point": "resid_post",
                     "offset": 0, "width": d}],
                "semvec_shims": [
                    {"name": "more-formal", "axis": "formality", "scale": 1.0,
                     "tags": ["style"]},
                    {"name": "impossible", "axis": "humor", "scale": 1.0},
                ]})
    with open(cfg_yaml, "w") as f:
        yaml.safe_dump(raw, f)
    cfg = Config.load(cfg_yaml)

    out_gguf = os.path.join(ROOT, "extended.gguf")
    export_mod.run(cfg, out_gguf)
    r = gguf.GGUFReader(out_gguf)
    fields = {fl.name for fl in r.fields.values()}
    tnames = {t.name for t in r.tensors}
    check("optional feature flagged",
          list(r.fields["therobot.features_optional"].contents()) == ["semvec"])
    check("semvec KVs present", {"therobot.semvec.version", "therobot.semvec.hash",
          "therobot.semvec.named_dim", "therobot.semvec.site_count"} <= fields)
    check("semvec hash KV matches spec",
          r.fields["therobot.semvec.hash"].contents() == spec.spec_hash())
    check("axes KV length", len(list(r.fields["therobot.semvec.axes"].contents())) == spec.named_dim)
    check("site meta KVs", r.fields["therobot.semvec.site.0.name"].contents() == "s0"
          and int(r.fields["therobot.semvec.site.0.layer"].contents()) == 2)
    check("proj/calib/overlay tensors packaged",
          {"robot.semvec.s0.proj", "robot.semvec.s0.calib",
           "robot.semvec.s0.overlay"} <= tnames)
    tmap = {t.name: t for t in r.tensors}
    # GGUF stores shapes reversed (ggml order) — compare as sets of dims
    check("packaged proj dims", sorted(tmap["robot.semvec.s0.proj"].shape) == sorted((d, spec.dim)))
    check("base tensor intact", "dummy.weight" in tnames)

    stripped = os.path.join(ROOT, "stripped.gguf")
    export_mod.strip(cfg, out_gguf, stripped)
    rs = gguf.GGUFReader(stripped)
    check("strip removes semvec tensors",
          not any(t.name.startswith("robot.") for t in rs.tensors))
    check("strip removes therobot KVs",
          not any(fl.name.startswith("therobot.") for fl in rs.fields.values()))

    # ---------------------------------------------- shim-compile (§4.5)
    print("shim-compile — semvec-defined, overlay-compiled")
    shimc.run(cfg)
    lock2 = _yaml.safe_load(open(os.path.join(ROOT, "cfg.lock.yaml")))
    sc = lock2["shim_compile"]
    check("formality shim admitted", any(s["name"] == "more-formal" for s in sc["admitted"]))
    check("unwritable axis rejected", any(s["name"] == "impossible" for s in sc["rejected"]))
    mod_path = os.path.join(cfg.workdir, "modules", "more-formal.gguf")
    rm = gguf.GGUFReader(mod_path)
    check("module arch", rm.fields["general.architecture"].contents() == "therobot-shim")
    check("module semvec provenance",
          rm.fields["therobot.shim.semvec.axis"].contents() == "formality" and
          rm.fields["therobot.shim.semvec.hash"].contents() == spec.spec_hash())
    steer = {t.name: t for t in rm.tensors}["robot.shim.steer"].data
    shift = np.asarray(steer, dtype=np.float32) @ proj
    check("compiled steer moves target by scale",
          abs(float(shift[fi]) - 1.0) < 1e-3, f"{float(shift[fi]):.4f}")
    reg = json.load(open(sc["registry"]))
    ent = next(s for s in reg["shims"] if s["name"] == "more-formal")
    check("registry semvec-keyed", ent["semvec"]["hash"] == spec.spec_hash()
          and ent["semvec"]["site"] == "s0")

print()
if FAILURES:
    print(f"EXTRACTION TESTS: FAILED ({len(FAILURES)}): {FAILURES}")
    sys.exit(1)
print("EXTRACTION TESTS: OK")
