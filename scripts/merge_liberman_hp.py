#!/usr/bin/env python3
"""Assemble scenario C HP JSON from T5 sklearn export + Colab MLP."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--t5-sklearn",
        type=Path,
        default=Path("figures/cache/stage2_best_hp/liberman_t5_sklearn.json"),
    )
    parser.add_argument(
        "--mlp",
        type=Path,
        default=Path("figures/cache/nn_colab_liberman/results/mlp_best_hp.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("figures/cache/stage2_best_hp/C.json"),
    )
    args = parser.parse_args()

    t5 = json.loads(args.t5_sklearn.read_text(encoding="utf-8"))
    mlp = json.loads(args.mlp.read_text(encoding="utf-8"))

    payload = {
        "scenario": "C",
        "config_id": t5.get("config_id", "T5"),
        "format": t5.get("format", "wide"),
        "noise_label": t5.get("noise_label", "predicted"),
        "RF": t5["RF"],
        "XGB": t5["XGB"],
        "MLP": mlp,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
