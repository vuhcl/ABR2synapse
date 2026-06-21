#!/usr/bin/env python3
"""Export print-ready paper Figures 6–8 (Stage-1 / Stage-2 MLP SHAP)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.paper_shap_figures import (  # noqa: E402
    export_all_paper_shap_figures,
    paper_shap_digest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage1-cache",
        type=Path,
        default=None,
        help="Override figures/cache/shap_stage1/",
    )
    parser.add_argument(
        "--mlp-cache",
        type=Path,
        default=None,
        help="Override figures/cache/shap_mlp/",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default figures/eda/)",
    )
    args = parser.parse_args()

    kw: dict = {}
    if args.stage1_cache is not None:
        kw["stage1_cache_dir"] = args.stage1_cache
    if args.mlp_cache is not None:
        kw["mlp_cache_dir"] = args.mlp_cache
    if args.out_dir is not None:
        kw["out_dir"] = args.out_dir

    print(
        paper_shap_digest(
            stage1_cache_dir=args.stage1_cache,
            mlp_cache_dir=args.mlp_cache,
        )
    )
    paths = export_all_paper_shap_figures(**kw)
    for name, base in paths.items():
        print(f"Wrote {name}: {base.with_suffix('.png')} + .svg")
    print("export_paper_shap_figures: done")


if __name__ == "__main__":
    main()
