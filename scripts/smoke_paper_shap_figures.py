#!/usr/bin/env python3
"""Smoke test for paper Figures 6–8 exports."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.abr_univariate_eda import FIG_EDA_DIR  # noqa: E402
from utils.paper_shap_figures import (  # noqa: E402
    FIGURE_06_STEM,
    FIGURE_07_STEM,
    FIGURE_08_STEM,
)

STEMS = (FIGURE_06_STEM, FIGURE_07_STEM, FIGURE_08_STEM)
CAPTIONS = ("figure_06_caption.txt", "figure_07_caption.txt", "figure_08_caption.txt")


def main() -> None:
    out = FIG_EDA_DIR
    for stem in STEMS:
        for ext in (".png", ".svg"):
            p = out / f"{stem}{ext}"
            if not p.is_file() or p.stat().st_size == 0:
                raise FileNotFoundError(f"Missing or empty: {p}")

    for name in CAPTIONS:
        p = out / name
        if not p.is_file() or p.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty caption: {p}")

    print("OK paper SHAP figures:", ", ".join(STEMS))
    print("OK captions:", ", ".join(CAPTIONS))
    print("smoke_paper_shap_figures: all checks passed")


if __name__ == "__main__":
    main()
