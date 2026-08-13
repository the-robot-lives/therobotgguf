"""R1 — Record: run the calibration corpus through the frozen donor with
forward hooks on candidate sites; store slices only, fp16. [needs the HF
stack; untested in-repo until a GPU/checkpoint environment runs it]

The recordings + weak labels are the training substrate for everything
downstream: probes (R2), thresholds (R4), shims (R5), and later accreted
modules. Weak labels come from configured labelers — label once, reuse
forever.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np

from .config import Config, Lockfile
from .recordings import RecordingStore


def _stored_windows(cfg: Config):
    tokens_path = os.path.join(cfg.recordings_dir, "tokens.npy")
    if not os.path.exists(tokens_path):
        raise SystemExit("recordings carry no tokens.npy — re-run `robotgguf record`")
    ids = np.load(tokens_path)
    window = int(np.load(os.path.join(cfg.recordings_dir, "window_size.npy"))[0])
    return ids, [ids[lo:lo + window].tolist() for lo in range(0, len(ids), window)]


def relabel(cfg: Config) -> None:
    """Regenerate the weak labels from the stored token ids without re-running
    the model (labelers change more often than recordings do). When the config
    names a semvec spec, this rebuilds labels/vector.npy at tier t0 and derives
    the categorical views from it; `robotgguf labelvec` layers t1/t2 on top."""
    try:
        from transformers import AutoTokenizer  # noqa: PLC0415
    except ImportError as e:
        raise SystemExit(f"relabel needs the HF tokenizer (pip install 'robotgguf[hf]'): {e}")

    ids, windows = _stored_windows(cfg)
    tok = AutoTokenizer.from_pretrained(cfg.donor)

    if cfg.semvec:
        labels = _build_vector_labels(cfg, tok, windows, n=len(ids))
    else:
        from .labelers import label_token_windows  # noqa: PLC0415
        labels = label_token_windows(tok, windows, cfg.attributes)
    for attr, y in labels.items():
        np.save(os.path.join(cfg.recordings_dir, "labels", f"{attr}.npy"), y[: len(ids)])
        print(f"relabel: labels[{attr}] regenerated ({len(np.unique(y))} classes present)")


def _pieces_per_window(tok, windows: list) -> list:
    return [[tok.decode([tid]) for tid in ids] for ids in windows]


def _build_vector_labels(cfg: Config, tok, windows: list, n: int) -> dict:
    """Assemble labels/vector.npy at tier t0 + the categorical views for
    cfg.attributes (extraction-v1 §3). Higher tiers (t1 classifiers/embedder,
    t2 teacher) overwrite axes later via `robotgguf labelvec` — same store,
    richer sources, no model re-run."""
    from . import labelers_t0, semvec  # noqa: PLC0415

    spec = semvec.SemvecSpec.load(cfg.resolve(cfg.semvec))
    pieces = _pieces_per_window(tok, windows)
    total = sum(len(p) for p in pieces)
    builder = semvec.VectorBuilder(spec, total)
    labelers_t0.fill(builder, spec, pieces)
    vec = builder.finish()[:n]

    os.makedirs(os.path.join(cfg.recordings_dir, "labels"), exist_ok=True)
    np.save(os.path.join(cfg.recordings_dir, "labels", "vector.npy"), vec)
    semvec.save_sources(cfg.recordings_dir, spec, builder.sources)
    print(f"labels: vector.npy [{len(vec)}, {spec.dim}] written "
          f"(semvec {spec.version}, {len(builder.sources)} axis(es) from t0)")
    return semvec.categorical_views(spec, vec, cfg.attributes)


def labelvec(cfg: Config, tiers: str = "t1", fit_basis_out: str = "",
             teacher_sample: int = 0) -> None:
    """Layer t1/t2 sources onto labels/vector.npy from stored tokens
    (extraction-v1 C2). Sentence-granular: unique sentences are labeled once
    and spread to positions. `--fit-basis` freezes the Block-B reduction."""
    try:
        from transformers import AutoTokenizer  # noqa: PLC0415
    except ImportError as e:
        raise SystemExit(f"labelvec needs the HF tokenizer (pip install 'robotgguf[hf]'): {e}")
    from . import labelers_t0, labelers_t1, semvec  # noqa: PLC0415

    if not cfg.semvec:
        raise SystemExit("labelvec: config names no `semvec:` spec")
    spec = semvec.SemvecSpec.load(cfg.resolve(cfg.semvec))
    ids, windows = _stored_windows(cfg)
    tok = AutoTokenizer.from_pretrained(cfg.donor)
    pieces = _pieces_per_window(tok, windows)

    # unique sentences + position→sentence index (label once, reuse forever)
    sents, sent_of_pos = [], []
    seen: dict = {}
    for pw in pieces:
        text = "".join(pw)
        spans = labelers_t0._sentence_spans(text)
        ci, si = 0, 0
        for piece in pw:
            mid = ci + max(1, len(piece)) // 2
            while si + 1 < len(spans) and mid >= spans[si][1]:
                si += 1
            a, b = spans[si]
            key = text[a:b]
            if key not in seen:
                seen[key] = len(sents)
                sents.append(key)
            sent_of_pos.append(seen[key])
            ci += len(piece)
    sent_of_pos = np.asarray(sent_of_pos[: len(ids)], dtype=np.int64)
    print(f"labelvec: {len(sents):,} unique sentence(s) over {len(sent_of_pos):,} positions")

    if fit_basis_out:
        emb = labelers_t1.embed_sentences(sents, spec)
        sha = semvec.fit_basis(emb, spec.latent_dim, cfg.resolve(fit_basis_out))
        print(f"labelvec: basis frozen at {fit_basis_out}\n"
              f"  pin in {os.path.basename(spec.path)}:  latent.basis: {fit_basis_out}\n"
              f"                                          latent.basis_sha256: {sha}")
        return

    vec_path = os.path.join(cfg.recordings_dir, "labels", "vector.npy")
    if not os.path.exists(vec_path):
        raise SystemExit("labelvec: no labels/vector.npy — run `robotgguf relabel` first")
    vec = np.load(vec_path).astype(np.float32)
    builder = semvec.VectorBuilder(spec, len(vec))
    builder.vec = vec
    src_path = os.path.join(cfg.recordings_dir, "labels", "vector_sources.json")
    if os.path.exists(src_path):
        import json  # noqa: PLC0415
        with open(src_path) as f:
            builder.sources = json.load(f).get("sources") or {}

    want = {t.strip() for t in tiers.split(",")}
    if "t1" in want:
        labelers_t1.fill_language(builder, spec, sents, sent_of_pos)
        labelers_t1.fill_weborganizer(builder, spec, sents, sent_of_pos)
        labelers_t1.fill_latent(builder, spec, sents, sent_of_pos, resolve=cfg.resolve)
    if "t2" in want:
        from . import labelers_teacher  # noqa: PLC0415
        rng = np.random.default_rng(7)
        k = min(teacher_sample or 100_000, len(sents))
        sample_idx = rng.choice(len(sents), size=k, replace=False)
        cache = labelers_teacher.AnnotationCache(
            os.path.join(cfg.workdir, "teacher-cache.jsonl"), spec)
        scores = labelers_teacher.annotate(spec, [sents[i] for i in sample_idx], cache)
        emb = labelers_t1.embed_sentences(sents, spec)
        w, r = labelers_teacher.distill(scores, emb[sample_idx])
        axes = labelers_teacher._axes_for_teacher(spec)
        pred = np.concatenate([emb, np.ones((len(emb), 1), dtype=np.float32)], axis=1) @ w
        for j, ax in enumerate(axes):
            print(f"labelvec: t2 {ax.name}: distill r={r[j]:.3f}")
            if r[j] >= 0.5:               # C2 bar: demote weak axes, never force
                builder.set_axis(ax.name, pred[sent_of_pos, j], tier="t2")
            else:
                print(f"labelvec:   → below bar, axis stays at its lower-tier source (finding)")

    out = builder.finish()
    np.save(vec_path, out)
    semvec.save_sources(cfg.recordings_dir, spec, builder.sources)
    # regenerate the categorical views from the richer vector
    views = semvec.categorical_views(spec, out, cfg.attributes)
    for attr, y in views.items():
        np.save(os.path.join(cfg.recordings_dir, "labels", f"{attr}.npy"), y)
    print(f"labelvec: vector.npy updated ({tiers}); categorical views regenerated")


def _decoder_layers(model):
    """Find the transformer block ModuleList regardless of wrapper layout
    (plain CausalLM, ConditionalGeneration, language_model nesting)."""
    import torch.nn as nn  # noqa: PLC0415
    for attr in ("model.layers", "model.model.layers",
                 "model.language_model.layers", "transformer.h"):
        obj = model
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
            if isinstance(obj, nn.ModuleList) and len(obj) >= 2:
                return obj
        except AttributeError:
            continue
    # fallback: largest ModuleList of decoder blocks
    best = None
    for _, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) >= 2:
            if best is None or len(mod) > len(best):
                best = mod
    if best is None:
        raise SystemExit("record: could not locate the decoder layer stack")
    return best


def run(cfg: Config, max_tokens: int = 200_000, window: int = 512) -> None:
    try:
        import torch  # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
    except ImportError as e:
        raise SystemExit(f"R1 needs the HF stack (pip install 'robotgguf[hf]'): {e}")

    lock = Lockfile(cfg.lockfile_path)
    survey = lock.section("survey")
    # candidate sites: prefer the survey (R0's measured output), fall back to
    # the config so record works without a full HF ingest pass
    site_list = survey.get("candidate_sites") or cfg.candidate_sites
    if not site_list:
        raise SystemExit("record: no candidate_sites in the lockfile survey or the config")
    sites = {s["name"]: {k: s[k] for k in ("layer", "point", "offset", "width")}
             for s in site_list}

    tok = AutoTokenizer.from_pretrained(cfg.donor)

    # device: prefer CUDA, then Apple MPS (the M-series GPU), else CPU. MPS
    # moves the dense matmuls (most of the cost) onto the GPU; any GDN op
    # without an MPS kernel falls back to CPU when PYTORCH_ENABLE_MPS_FALLBACK
    # is set (we set it below), so the run never errors — it just runs those
    # ops on CPU. Set ROBOT_DEVICE=cpu to force CPU.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    forced = os.environ.get("ROBOT_DEVICE", "").strip().lower()
    if forced in ("cpu", "mps", "cuda"):
        device = forced
    elif torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # fp16 on GPU (MPS/CUDA fp16 is well supported); bf16 on CPU for range
    dtype = torch.float16 if device in ("cuda", "mps") else torch.bfloat16
    kw = dict(low_cpu_mem_usage=True)

    def _load(cls):
        try:
            return cls.from_pretrained(cfg.donor, dtype=dtype, **kw)
        except TypeError:  # older transformers use torch_dtype
            return cls.from_pretrained(cfg.donor, torch_dtype=dtype, **kw)

    try:
        model = _load(AutoModelForCausalLM)
    except ValueError:
        # multimodal wrappers (Qwen3_5MoeForConditionalGeneration etc.) don't
        # map under CausalLM; the text stack still nests inside — layer
        # discovery below handles the wrapper
        from transformers import AutoModelForImageTextToText  # noqa: PLC0415
        model = _load(AutoModelForImageTextToText)
    model.eval().to(device)
    print(f"record: device={device} dtype={dtype}")

    # stratified corpus loading (extraction-v1 C1 — replaces the v0
    # head-to-tail read that drew everything from the first file's start)
    from . import corpus as corpus_mod  # noqa: PLC0415
    strata = corpus_mod.load_strata(cfg)
    win_ids, win_domains, domains = corpus_mod.stratified_windows(
        strata, tok, max_tokens, window)
    ids = [t for w in win_ids for t in w]
    corpus_hash = hashlib.sha256(
        (str(ids[:1024]) + corpus_mod.strata_hash(strata)).encode()).hexdigest()[:16]
    shares = corpus_mod.domain_shares_report(win_domains, domains)
    print(f"record: {len(win_ids)} window(s), per-stratum shares: {shares}")

    # forward hooks capture the residual stream after each covered block
    captured: dict[str, list] = {name: [] for name in sites}
    layers = _decoder_layers(model)  # robust to wrapper layouts (Qwen3.5 etc.)

    def make_hook(name, off, width):
        def hook(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured[name].append(h[0, :, off:off + width].detach().float().to(torch.float16).cpu())
        return hook

    handles = [layers[s["layer"]].register_forward_hook(
                   make_hook(n, s["offset"], s["width"]))
               for n, s in sites.items()]

    import gc, sys, time  # noqa: PLC0415
    total = len(win_ids) * window
    print(f"record: {len(win_ids)} forward window(s) of {window} tokens on {device}", file=sys.stderr, flush=True)
    t0 = time.time()
    with torch.no_grad():
        for wi, chunk_ids in enumerate(win_ids):
            # use_cache=False: recording is stateless per window; the GDN/attn
            # cache would otherwise accumulate and OOM on long corpora
            model(torch.tensor([chunk_ids], device=device), use_cache=False)
            gc.collect()
            if wi % 5 == 0 or wi == len(win_ids) - 1:
                done = (wi + 1) * window
                el = time.time() - t0
                rate = done / el if el > 0 else 0.0
                eta = (total - done) / rate if rate > 0 else 0.0
                print(f"\rrecord: forward {done:,}/{total:,} tokens "
                      f"({100.0*done/total:.0f}%)  {rate:.0f} tok/s  ETA {eta/60:.1f} min   ",
                      end="", file=sys.stderr, flush=True)
    print(f"\rrecord: forward pass done — {total:,} tokens in {(time.time()-t0)/60:.1f} min"
          + " " * 20, file=sys.stderr, flush=True)
    for h in handles:
        h.remove()

    acts = {n: torch.cat(c, dim=0).numpy() for n, c in captured.items()}
    n_samples = len(next(iter(acts.values())))

    # keep the token ids first so label passes can regenerate without the model
    os.makedirs(cfg.recordings_dir, exist_ok=True)
    np.save(os.path.join(cfg.recordings_dir, "tokens.npy"),
            np.concatenate([np.asarray(w, dtype=np.int64) for w in win_ids]))
    np.save(os.path.join(cfg.recordings_dir, "window_size.npy"),
            np.asarray([window], dtype=np.int64))

    # labels over the same token stream, sentence-granular. semvec configs get
    # the t0 label vector + categorical views; v0 configs keep the heuristics.
    if cfg.semvec:
        labels = _build_vector_labels(cfg, tok, win_ids, n=n_samples)
        labeler = "semvec-t0"
    else:
        from .labelers import label_token_windows  # noqa: PLC0415
        labels = label_token_windows(tok, win_ids, cfg.attributes)
        labeler = "heuristic-v0"
    # the domain id is a free, exact label and the cross-domain admission key
    labels["domain"] = np.repeat(win_domains, window)[:n_samples]

    store = RecordingStore(cfg.recordings_dir)
    store.write(model=cfg.config_hash(), corpus=corpus_hash, sites=sites,
                acts=acts, labels=labels, domain_names=domains)

    lock.update("record", {"n_samples": int(n_samples),
                           "corpus_hash": corpus_hash,
                           "sites": sorted(sites),
                           "domain_shares": shares,
                           "labeler": labeler})
    for attr in cfg.attributes:
        vals, counts = np.unique(labels[attr], return_counts=True)
        dist = ", ".join(f"{v}:{c}" for v, c in zip(vals.tolist(), counts.tolist()))
        print(f"record: labels[{attr}] class counts: {dist}")
    print(f"record: {len(acts)} site(s) recorded to {cfg.recordings_dir}")
