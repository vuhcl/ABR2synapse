#!/usr/bin/env python3
"""HP-tune mlp/cnn/cnn_full on Liberman Colab pack; export comparison parquets."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.nn_colab_export import DEFAULT_OUT, export_liberman_nn_colab_pack
from utils.nn_colab_train import run_liberman_nn_hp_comparison


def main() -> None:
    pack_dir = export_liberman_nn_colab_pack(DEFAULT_OUT)
    out_dir = pack_dir / "results"
    summary, nn_results = run_liberman_nn_hp_comparison(pack_dir, out_dir, verbose=True)
    cache = Path("figures/cache")
    cache.mkdir(parents=True, exist_ok=True)
    nn_results.to_parquet(cache / "liberman_nn_hp_comparison.parquet", index=False)
    summary.to_parquet(cache / "liberman_nn_hp_summary.parquet", index=False)
    best = nn_results.loc[nn_results["r2_test"].idxmax()]
    print(nn_results.sort_values("r2_test", ascending=False).to_string(index=False))
    print(
        f"\nBest NN: {best['model']} ({best['config_id']}) "
        f"R²={best['r2_test']:.3f} RMSE={best['rmse_test']:.3f}"
    )


if __name__ == "__main__":
    main()
