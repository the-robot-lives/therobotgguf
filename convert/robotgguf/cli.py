"""robotgguf CLI — `robotgguf <stage> --config configs/<donor>.yaml`."""
from __future__ import annotations

import argparse

from .config import Config


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="robotgguf",
                                description="Retrofit LLM checkpoints onto the therobot runtime")
    p.add_argument("--config", required=True, help="per-donor YAML config")
    sub = p.add_subparsers(dest="stage", required=True)

    sub.add_parser("ingest", help="R0 — survey the donor checkpoint [HF stack]")
    r1 = sub.add_parser("record", help="R1 — record candidate cleave sites [HF stack]")
    r1.add_argument("--max-tokens", type=int, default=200_000)
    sub.add_parser("relabel", help="R1.5 — regenerate labels (t0 vector + views when semvec) from stored tokens [HF tokenizer]")
    lv = sub.add_parser("labelvec", help="R1.5 — layer t1/t2 semvec sources onto labels/vector.npy")
    lv.add_argument("--tiers", default="t1", help="comma list: t1 (classifiers+embedder), t2 (teacher)")
    lv.add_argument("--fit-basis", default="", metavar="OUT",
                    help="fit + freeze the Block-B reduction to OUT (.npz) and exit")
    lv.add_argument("--teacher-sample", type=int, default=0,
                    help="teacher sample size for t2 (default 100k)")
    sub.add_parser("cleave", help="R2 — probe training + bottleneck selection")
    r3 = sub.add_parser("graft", help="R3 — leaky state + modulator graft")
    r3.add_argument("--steps", type=int, default=0,
                    help="training steps (0 = emit function-preserving init only)")
    sub.add_parser("calibrate", help="R4 — delta thresholds from recordings")
    sub.add_parser("shims", help="R5 — steering shims + admission + registry")
    sub.add_parser("shim-compile", help="compile semvec-defined shims for this donor (extraction-v1 §4.5)")
    sub.add_parser("settle", help="R6 — settle-track configuration")
    r7 = sub.add_parser("export", help="R7 — emit the extended GGUF")
    r7.add_argument("--out", required=True)
    r7s = sub.add_parser("strip", help="R7 — downgrade an extended GGUF to stock")
    r7s.add_argument("input")
    r7s.add_argument("output")
    r8 = sub.add_parser("verify", help="R8 — drive the runtime gates")
    r8.add_argument("--parity-bin", default=None,
                    help="parity test binary (compares base_gguf vs export)")

    args = p.parse_args(argv)
    cfg = Config.load(args.config)

    if args.stage == "ingest":
        from . import ingest; ingest.run(cfg)
    elif args.stage == "record":
        from . import record; record.run(cfg, max_tokens=args.max_tokens)
    elif args.stage == "relabel":
        from . import record; record.relabel(cfg)
    elif args.stage == "labelvec":
        from . import record
        record.labelvec(cfg, tiers=args.tiers, fit_basis_out=args.fit_basis,
                        teacher_sample=args.teacher_sample)
    elif args.stage == "cleave":
        from . import cleave; cleave.run(cfg)
    elif args.stage == "graft":
        from . import graft; graft.run(cfg, steps=args.steps)
    elif args.stage == "calibrate":
        from . import calibrate; calibrate.run(cfg)
    elif args.stage == "shims":
        from . import shims; shims.run(cfg)
    elif args.stage == "shim-compile":
        from . import shimc; shimc.run(cfg)
    elif args.stage == "settle":
        from . import settle; settle.run(cfg)
    elif args.stage == "export":
        from . import export; export.run(cfg, args.out)
    elif args.stage == "strip":
        from . import export; export.strip(cfg, args.input, args.output)
    elif args.stage == "verify":
        from . import verify; verify.run(cfg, parity_bin=args.parity_bin)


if __name__ == "__main__":
    main()
