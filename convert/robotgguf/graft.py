"""R3 — Graft leaky state (§B) + modulator/FiLM (§A) onto the frozen donor.
[needs the HF stack; untested in-repo until a GPU environment runs it]

Every graft is function-preserving at insertion (006's nascent-unit trick):
out_proj zero, γ-bias one, everything else zero — a grafted model at init is
bit-for-bit the donor. Training touches new parameters only (the core is
frozen), anchored by KL to the donor's own logits so the grafts learn to help
without drifting the base distribution. Trained tensors land as .npy files
the R7 exporter picks up; if this stage never runs, R7 exports zero grafts
and the file is a function-preserving L2.
"""
from __future__ import annotations

import os

import numpy as np

from .config import Config, Lockfile


def _n_embd_from_base(cfg: Config) -> int:
    """Read the donor's embedding length from the base GGUF, so the zero-graft
    path works without an HF `ingest` pass."""
    if not cfg.base_gguf:
        return 0
    try:
        gguf = cfg.gguf_module()
        reader = gguf.GGUFReader(cfg.resolve(cfg.base_gguf))
        field = reader.fields.get(f"{cfg.base_architecture}.embedding_length")
        return int(field.contents()) if field is not None else 0
    except Exception:
        return 0


def run(cfg: Config, steps: int = 0, lr: float = 1e-4) -> None:
    lock = Lockfile(cfg.lockfile_path)
    tensor_dir = os.path.join(cfg.workdir, "grafts")
    os.makedirs(tensor_dir, exist_ok=True)

    survey = lock.section("survey")
    n_embd = int(survey.get("n_embd", 0)) or _n_embd_from_base(cfg)
    s_width = sum(int(b["width"]) for b in cfg.state_banks)
    m_dim = int(cfg.modulator["dim"])
    film_layers = list(cfg.raw.get("film_layers", cfg.state_layers))

    if steps == 0:
        # zero-effect graft: emit function-preserving init tensors only.
        # This path needs no torch and keeps the export honest — the runtime's
        # parity gate will prove the graft contributed nothing.
        if n_embd == 0:
            raise SystemExit("graft: could not determine n_embd — run `robotgguf ingest` "
                             "or set base_gguf in the config")
        rng = np.random.default_rng(0)
        for layer in cfg.state_layers:
            np.save(os.path.join(tensor_dir, f"blk.{layer}.robot_state.alpha.npy"),
                    np.zeros(s_width, dtype=np.float32))  # σ(0)=0.5 mixed decay
            np.save(os.path.join(tensor_dir, f"blk.{layer}.robot_state.in_proj.weight.npy"),
                    (rng.standard_normal((s_width, n_embd)) * 0.02).astype(np.float32))
            np.save(os.path.join(tensor_dir, f"blk.{layer}.robot_state.out_proj.weight.npy"),
                    np.zeros((n_embd, s_width), dtype=np.float32))  # zero at graft
        np.save(os.path.join(tensor_dir, "robot.mod.alpha.npy"), np.zeros(m_dim, dtype=np.float32))
        pool_in = n_embd if cfg.modulator.get("source") != "glacial" else \
            sum(int(b["width"]) for b in cfg.state_banks if b["name"] == "glacial")
        np.save(os.path.join(tensor_dir, "robot.mod.pool.weight.npy"),
                (rng.standard_normal((m_dim, pool_in)) * 0.02).astype(np.float32))
        np.save(os.path.join(tensor_dir, "robot.mod.cell.weight.npy"),
                np.zeros((m_dim, m_dim), dtype=np.float32))
        for layer in film_layers:
            np.save(os.path.join(tensor_dir, f"blk.{layer}.robot_film.gamma.weight.npy"),
                    np.zeros((n_embd, m_dim), dtype=np.float32))
            np.save(os.path.join(tensor_dir, f"blk.{layer}.robot_film.gamma.bias.npy"),
                    np.ones(n_embd, dtype=np.float32))  # identity FiLM
            np.save(os.path.join(tensor_dir, f"blk.{layer}.robot_film.beta.weight.npy"),
                    np.zeros((n_embd, m_dim), dtype=np.float32))
        lock.update("graft", {"trained": False, "steps": 0,
                              "tensor_dir": tensor_dir,
                              "film_layers": film_layers})
        print(f"graft: function-preserving init emitted ({tensor_dir}); "
              f"train with --steps N on a GPU host")
        return

    # ---- trained graft (GPU path) ----
    try:
        import torch  # noqa: PLC0415
        import torch.nn.functional as F  # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
    except ImportError as e:
        raise SystemExit(f"R3 training needs the HF stack (pip install 'robotgguf[hf]'): {e}")

    torch.manual_seed(0)
    donor = AutoModelForCausalLM.from_pretrained(cfg.donor, torch_dtype=torch.float32)
    donor.eval()
    for p in donor.parameters():
        p.requires_grad_(False)  # the core is frozen, permanently

    layers = donor.model.layers
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    donor.to(dev)

    class Graft(torch.nn.Module):
        """Leaky state + FiLM around covered blocks; forward mirrors the
        runtime's per-position scan exactly (llama-robot-model.cpp)."""

        def __init__(self):
            super().__init__()
            self.alpha = torch.nn.ParameterDict()
            self.in_proj = torch.nn.ModuleDict()
            self.out_proj = torch.nn.ModuleDict()
            self.film_g = torch.nn.ModuleDict()
            self.film_b = torch.nn.ModuleDict()
            for L in cfg.state_layers:
                k = str(L)
                self.alpha[k] = torch.nn.Parameter(torch.zeros(s_width))
                self.in_proj[k] = torch.nn.Linear(n_embd, s_width, bias=False)
                op = torch.nn.Linear(s_width, n_embd, bias=False)
                torch.nn.init.zeros_(op.weight)  # function-preserving
                self.out_proj[k] = op
            for L in film_layers:
                k = str(L)
                g = torch.nn.Linear(m_dim, n_embd)
                torch.nn.init.zeros_(g.weight); torch.nn.init.ones_(g.bias)
                b = torch.nn.Linear(m_dim, n_embd, bias=False)
                torch.nn.init.zeros_(b.weight)
                self.film_g[k], self.film_b[k] = g, b

        def state_step(self, k, h):  # h: [B, T, n_embd]
            a = torch.sigmoid(self.alpha[k])
            f = self.in_proj[k](h)
            s = torch.zeros(h.shape[0], s_width, device=h.device)
            outs = []
            for t in range(h.shape[1]):
                s = a * s + (1 - a) * f[:, t]
                outs.append(s)
            return h + self.out_proj[k](torch.stack(outs, dim=1))

    graft = Graft().to(dev)
    hooks = []
    m_vec = torch.zeros(1, m_dim, device=dev)

    def state_hook(k):
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            h2 = graft.state_step(k, h)
            if k in graft.film_g:
                h2 = graft.film_g[k](m_vec).unsqueeze(1) * h2 + \
                     graft.film_b[k](m_vec).unsqueeze(1)
            return (h2,) + out[1:] if isinstance(out, tuple) else h2
        return hook

    for L in sorted(set(map(int, cfg.state_layers)) | set(map(int, film_layers))):
        hooks.append(layers[L].register_forward_hook(state_hook(str(L))))

    tok = AutoTokenizer.from_pretrained(cfg.donor)
    ids = []
    for spec in cfg.corpus:
        with open(cfg.resolve(spec)) as f:
            ids.extend(tok(f.read()).input_ids)
    opt = torch.optim.AdamW(graft.parameters(), lr=lr)

    window = 256
    for step in range(steps):
        lo = (step * window) % max(1, len(ids) - window - 1)
        chunk = torch.tensor([ids[lo:lo + window]], device=dev)
        with torch.no_grad():
            for h in hooks: h.remove()
            ref = donor(chunk).logits            # frozen-donor anchor
            hooks = [layers[L].register_forward_hook(state_hook(str(L)))
                     for L in sorted(set(map(int, cfg.state_layers)) | set(map(int, film_layers)))]
        out = donor(chunk).logits
        loss = F.kl_div(F.log_softmax(out, -1), F.softmax(ref, -1), reduction="batchmean")
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0:
            print(f"graft: step {step} distill KL {loss.item():.5f}")

    for h in hooks:
        h.remove()

    # dump trained tensors in the exporter's layout
    for L in cfg.state_layers:
        k = str(L)
        np.save(os.path.join(tensor_dir, f"blk.{L}.robot_state.alpha.npy"),
                graft.alpha[k].detach().cpu().numpy())
        np.save(os.path.join(tensor_dir, f"blk.{L}.robot_state.in_proj.weight.npy"),
                graft.in_proj[k].weight.detach().cpu().numpy())
        np.save(os.path.join(tensor_dir, f"blk.{L}.robot_state.out_proj.weight.npy"),
                graft.out_proj[k].weight.detach().cpu().numpy())
    for L in film_layers:
        k = str(L)
        np.save(os.path.join(tensor_dir, f"blk.{L}.robot_film.gamma.weight.npy"),
                graft.film_g[k].weight.detach().cpu().numpy())
        np.save(os.path.join(tensor_dir, f"blk.{L}.robot_film.gamma.bias.npy"),
                graft.film_g[k].bias.detach().cpu().numpy())
        np.save(os.path.join(tensor_dir, f"blk.{L}.robot_film.beta.weight.npy"),
                graft.film_b[k].weight.detach().cpu().numpy())

    lock.update("graft", {"trained": True, "steps": steps,
                          "tensor_dir": tensor_dir,
                          "film_layers": film_layers})
    print(f"graft: trained {steps} step(s); tensors in {tensor_dir}")
