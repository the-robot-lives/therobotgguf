"""T1 — open classifiers + the embedder (extraction-v1 §3.3).

Fills semvec axes from models that already exist:
  - the sentence embedder → Block B (the latent block), through the FROZEN
    reduction (semvec.load_basis / reduce_embeddings — the same map queries
    use at runtime);
  - WebOrganizer topic/format classifiers → soft memberships projected onto
    the topic/discourse axes;
  - GlotLID/fastText → language axes sharpened from script to language id.

Heavy deps (torch, transformers, sentence-transformers, fasttext) are
imported lazily; every entry point degrades to a clear SystemExit naming the
missing extra. Runs on the Modal GPU app (tools/modal_record.py) or any host
with `pip install 'robotgguf[label]'`.
"""
from __future__ import annotations

import sys

import numpy as np

from .semvec import SemvecSpec, VectorBuilder, load_basis, reduce_embeddings

# WebOrganizer 24-way topic classes → semvec topic axes (extraction-v1 §7
# "label projection error" — the C2 κ holdout scores THESE projected labels).
# Classes absent here intentionally contribute nothing.
WEBORG_TOPIC_TO_AXIS = {
    "Software": "tech_software", "Software Dev.": "tech_software",
    "Hardware": "tech_hardware", "Electronics": "tech_hardware",
    "Science & Tech.": "science_phys", "Science, Tech. & Math": "science_phys",
    "Health": "medicine", "Medicine": "medicine",
    "Politics": "politics", "Finance & Business": "economics_finance",
    "Crime & Law": "law", "Education & Jobs": "education",
    "Sports & Fitness": "sports", "Art & Design": "arts",
    "Music": "music", "Literature": "literature_topic",
    "Movies": "film_tv", "Video Games": "games", "Games": "games",
    "Food & Dining": "food", "Travel": "travel", "Fashion & Beauty": "fashion",
    "Religion": "religion", "History": "history",
    "Home & Hobbies": "diy_crafts", "Transportation": "automotive",
    "Social Life": "relationships", "Entertainment": "film_tv",
}


def embed_sentences(sentences: list, spec: SemvecSpec, batch: int = 256) -> np.ndarray:
    """Embed with the spec-pinned embedder. Returns [N, E] float32."""
    lat = spec.latent or {}
    model_id = lat.get("embedder")
    if not model_id:
        raise SystemExit("t1: spec pins no embedder (latent.embedder is null)")
    try:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
    except ImportError:
        raise SystemExit("t1: pip install 'robotgguf[label]' (sentence-transformers)")
    model = SentenceTransformer(model_id, revision=lat.get("embedder_revision"),
                                trust_remote_code=True)
    out = model.encode(sentences, batch_size=batch, show_progress_bar=True,
                       convert_to_numpy=True, normalize_embeddings=False)
    return np.asarray(out, dtype=np.float32)


def fill_latent(builder: VectorBuilder, spec: SemvecSpec, sentences: list,
                sent_of_pos: np.ndarray, resolve=lambda p: p) -> None:
    """Embed unique sentences, reduce through the frozen basis, spread to
    positions. Refuses when the basis isn't frozen — an unpinned latent block
    would not be the standard, just noise that LOOKS like it."""
    basis = load_basis(spec, resolve)
    if basis is None:
        print("t1: latent basis not frozen (spec latent.basis is null) — "
              "skipping Block B; run `robotgguf labelvec --fit-basis` first",
              file=sys.stderr)
        return
    emb = embed_sentences(sentences, spec)
    red = reduce_embeddings(emb, basis)                     # [S, latent_dim]
    builder.set_latent(red[sent_of_pos], tier="t1")


def fill_weborganizer(builder: VectorBuilder, spec: SemvecSpec, sentences: list,
                      sent_of_pos: np.ndarray, batch: int = 128) -> None:
    """Topic classifier softmax → soft memberships on semvec topic axes,
    scaled to the ordinal range."""
    try:
        import torch  # noqa: PLC0415
        from transformers import (AutoModelForSequenceClassification,  # noqa: PLC0415
                                  AutoTokenizer)
    except ImportError:
        raise SystemExit("t1: pip install 'robotgguf[label]' (transformers+torch)")
    model_id = "WebOrganizer/TopicClassifier"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, trust_remote_code=True).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    id2label = model.config.id2label
    probs = np.zeros((len(sentences), len(id2label)), dtype=np.float32)
    with torch.no_grad():
        for lo in range(0, len(sentences), batch):
            enc = tok(sentences[lo:lo + batch], truncation=True, max_length=512,
                      padding=True, return_tensors="pt").to(device)
            p = torch.softmax(model(**enc).logits, dim=-1)
            probs[lo:lo + len(p)] = p.float().cpu().numpy()

    lo_s, hi_s = spec.scale
    hit = 0
    for ci, label in id2label.items():
        axis = WEBORG_TOPIC_TO_AXIS.get(label)
        if axis and axis in spec.by_name:
            builder.set_axis(axis, (probs[:, int(ci)] * hi_s)[sent_of_pos], tier="t1")
            hit += 1
    print(f"t1: weborganizer topic → {hit} axis(es) filled", file=sys.stderr)


def fill_language(builder: VectorBuilder, spec: SemvecSpec, sentences: list,
                  sent_of_pos: np.ndarray) -> None:
    """GlotLID language id → script axes (exact), recorded as t1 so it
    outranks the t0 char-class heuristic."""
    try:
        import fasttext  # noqa: PLC0415
        from huggingface_hub import hf_hub_download  # noqa: PLC0415
    except ImportError:
        raise SystemExit("t1: pip install 'robotgguf[label]' (fasttext + huggingface_hub)")
    path = hf_hub_download("cis-lmu/glotlid", "model.bin")
    model = fasttext.load_model(path)
    script_axis = {"Latn": "latin_script", "Jpan": "cjk_script",
                   "Hans": "cjk_script", "Hant": "cjk_script",
                   "Hang": "cjk_script", "Cyrl": "cyrillic_script"}
    cols = {a: np.zeros(len(sentences), dtype=np.float32)
            for a in ("latin_script", "cjk_script", "cyrillic_script", "other_script")}
    for i, s in enumerate(sentences):
        lab, _ = model.predict(s.replace("\n", " ")[:400])
        script = lab[0].rsplit("_", 1)[-1] if lab else "Latn"
        cols[script_axis.get(script, "other_script")][i] = 4.0
    for a, v in cols.items():
        if a in spec.by_name:
            builder.set_axis(a, v[sent_of_pos], tier="t1")
