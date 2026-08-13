#!/usr/bin/env python3
"""Generate candidate_sites blocks (extraction-v1 §4.2) instead of
hand-writing them per donor.

Survey grid from architecture facts:
  python3 tools/gen_sites.py survey --layers 40 --hidden 2048 [--width 256] [--every 4]
  → resid sites at L ≡ 2 (mod every), skipping block 0 and the final block

Focused grid from a survey run's winners:
  python3 tools/gen_sites.py focused --lockfile configs/<donor>.lock.yaml \
      [--top 4] [--widths 64,128,256] [--points resid_post,attn_out,ffn_out]
  → re-record the top-N sites (by named-axis admissions) across
    points × offsets × widths (offsets tile the hidden size by width)

Both print YAML ready to paste into (or `>>` onto) the donor config.
"""
import argparse
import sys

import yaml


def survey(args) -> list:
    sites = []
    for layer in range(args.every // 2, args.layers - 1, args.every):
        if layer == 0:
            continue
        sites.append({"name": f"resid{layer}", "layer": layer,
                      "point": "resid_post", "offset": 0, "width": args.width})
    return sites


def focused(args) -> list:
    with open(args.lockfile) as f:
        lock = yaml.safe_load(f)
    cvec = lock.get("cleave_vec") or {}
    ranked = sorted(cvec.get("sites", {}).items(),
                    key=lambda kv: -kv[1].get("n_admitted_named", 0))[: args.top]
    if not ranked:
        sys.exit("gen_sites: lockfile has no cleave_vec site results")
    widths = [int(w) for w in args.widths.split(",")]
    points = args.points.split(",")
    # widest width defines the recording; narrower probes slice it for free —
    # so emit one site per (layer, point, offset) at max width only
    w = max(widths)
    sites = []
    seen_layers = set()
    for sname, _info in ranked:
        layer = int("".join(ch for ch in sname if ch.isdigit()))
        if layer in seen_layers:
            continue
        seen_layers.add(layer)
        for point in points:
            for off in range(0, args.hidden, w):
                tag = {"resid_post": "resid", "attn_out": "attn", "ffn_out": "ffn"}.get(point, point)
                sites.append({"name": f"{tag}{layer}o{off}", "layer": layer,
                              "point": point, "offset": off, "width": w})
    return sites


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("survey")
    s.add_argument("--layers", type=int, required=True)
    s.add_argument("--hidden", type=int, required=True)
    s.add_argument("--width", type=int, default=256)
    s.add_argument("--every", type=int, default=4)
    f = sub.add_parser("focused")
    f.add_argument("--lockfile", required=True)
    f.add_argument("--hidden", type=int, required=True)
    f.add_argument("--top", type=int, default=4)
    f.add_argument("--widths", default="64,128,256")
    f.add_argument("--points", default="resid_post,attn_out,ffn_out")
    args = p.parse_args()

    sites = survey(args) if args.mode == "survey" else focused(args)
    bytes_per_tok = sum(s["width"] for s in sites) * 2
    print(f"# {args.mode} grid — {len(sites)} site(s), "
          f"{bytes_per_tok / 1024:.1f} KB/token recorded", file=sys.stderr)
    print(yaml.safe_dump({"candidate_sites": sites}, sort_keys=False))


if __name__ == "__main__":
    main()
