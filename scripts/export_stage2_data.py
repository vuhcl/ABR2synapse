#!/usr/bin/env python3
"""Export Stage-2 HP tuning parquet bundles (optional CLI)."""
from __future__ import annotations

from pathlib import Path

from utils.stage2_export import export_stage2_data


def main() -> None:
    out = export_stage2_data()
    for k, p in out.items():
        print(f"Wrote {k}: {p}")


if __name__ == "__main__":
    main()
