"""R0 — Ingest & survey the donor checkpoint. [needs the HF stack; untested
in-repo until a GPU/checkpoint environment runs it]

Loads the HF model, emits the survey section of the lockfile: architecture,
layer count, dims, candidate cleave sites (default: residual stream after
each block's FFN, every 4 blocks), tokenizer facts, licensing note; sanity-
runs generation.
"""
from __future__ import annotations

from .config import Config, Lockfile


def run(cfg: Config) -> None:
    try:
        import torch  # noqa: PLC0415
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
    except ImportError as e:
        raise SystemExit(f"R0 needs the HF stack (pip install 'robotgguf[hf]'): {e}")

    lock = Lockfile(cfg.lockfile_path)

    hf_cfg = AutoConfig.from_pretrained(cfg.donor)
    tok = AutoTokenizer.from_pretrained(cfg.donor)
    model = AutoModelForCausalLM.from_pretrained(cfg.donor, torch_dtype=torch.float16)
    model.eval()

    n_layer = int(getattr(hf_cfg, "num_hidden_layers"))
    n_embd = int(getattr(hf_cfg, "hidden_size"))

    # default candidate sites: a residual slice after each block's FFN, every
    # 4 blocks, avoiding block 0 and the final block (runtime restrictions)
    candidates = cfg.candidate_sites or [
        {"name": f"resid{layer}", "layer": layer, "point": "resid_post",
         "offset": 0, "width": min(128, n_embd)}
        for layer in range(2, n_layer - 1, 4)
    ]

    # sanity generation
    with torch.no_grad():
        ids = tok("The quick brown fox", return_tensors="pt").input_ids
        out = model.generate(ids, max_new_tokens=8, do_sample=False)
    sample = tok.decode(out[0], skip_special_tokens=True)

    lock.update("survey", {
        "donor": cfg.donor,
        "architecture": hf_cfg.model_type,
        "n_layer": n_layer,
        "n_embd": n_embd,
        "n_vocab": int(getattr(hf_cfg, "vocab_size")),
        "candidate_sites": candidates,
        "tokenizer": {"class": type(tok).__name__,
                      "bos": tok.bos_token_id, "eos": tok.eos_token_id},
        "license": getattr(hf_cfg, "license", None) or "check the model card",
        "sanity_generation": sample,
    })
    print(f"ingest: {cfg.donor} — {hf_cfg.model_type}, {n_layer} layers, "
          f"n_embd {n_embd}, {len(candidates)} candidate site(s)")
