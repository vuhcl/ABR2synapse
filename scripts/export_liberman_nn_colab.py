#!/usr/bin/env python3
"""Export Liberman NN Colab training pack (run from repo root)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.nn_colab_export import DEFAULT_OUT, export_liberman_nn_colab_pack

if __name__ == "__main__":
    export_liberman_nn_colab_pack(DEFAULT_OUT)
