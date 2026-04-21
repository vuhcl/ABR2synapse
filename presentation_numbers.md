# Presentation numbers (for slides & script)

Sources: Buran et al. 2025 (JARO); `abr_nn_stage2.ipynb` outputs; `abr_wide_long_comparison.ipynb` outputs; `presentation_benchmarks.ipynb` table; `figures/cache/stage1_wide_rf_metrics.json`. Liberman wide/long counts match `load_nn_stage2_data()` splits as printed in `abr_nn_stage2.ipynb` (Scenario A / Lib train unless noted).

---

## Dataset — Buran (Brad)

| Item | Value |
|------|--------|
| Animals (paper) | **57** CBA/CaJ (25 M, 32 F) |
| Cohorts | Young **17**; acute noise **13**; aged **14**; aged+noise **13** |
| ABR grid (paper) | **7** carrier freqs (5.6–45.2 kHz, half-octave); **15** SPLs (10–80 dB, 5 dB steps) |
| **Analysis pipeline** (wide / long, from `abr_nn_stage2` summary) | Wide **283** rows, **77** animals; long train e.g. Scenario A **2811** rows (varies by scenario B/C) |

---

## Dataset — Liberman (Wu et al. / WPZ)

| Item | Value |
|------|--------|
| **Analysis pipeline** (from `abr_nn_stage2` summary) | Wide **491** rows, **125** animals; long train Scenario A **3554** rows |
| TSV example (`WPZ100` ABR 8 kHz) | SPL columns include **15, 20, 25, 30, 35, 40, 50, 60, 70, 80** dB on that file (grid can omit some steps; odd dB coverage uneven across animals — motivates **even-SPL** benchmark variant in comparison notebook) |

*Exact `load_data()` row count after peak-finding was not re-run here (full pass is slow); use notebook one-liner if you need live counts.*

---

## Stage 1 — Wide RF noise classifier (`stage1_wide_rf_metrics.json`)

Evaluated separately per lab’s wide train / test framing in the pipeline that produced this file:

| Lab | CV acc (best search) | Animal-level test acc | Animal-level AUC |
|-----|----------------------|------------------------|------------------|
| **Brad** | **0.629** | **0.750** | **1.000** |
| **Liberman** | **0.601** | **0.857** | **0.891** |

*Use **Liberman** row for the main spoken line if you show a single line; mention **Brad** row if you discuss Buran-only training.*

---

## OLS baselines (`abr_wide_long_comparison.ipynb` printed outputs)

### Brad (train on Brad; metrics as in notebook output)

| Model | R² | RMSE |
|-------|-----|------|
| OLS baseline (amplitude @ 80 dB only) | **0.374** | **3.756** |
| OLS full (noise_preds + all features) | **0.022** | **4.695** |

### Liberman (train on Liberman)

| Model | R² | RMSE |
|-------|-----|------|
| OLS baseline (amplitude @ 80) | **0.229** | **3.090** |
| OLS full (noise_cat + all features) | **0.426** | **2.667** |

---

## Benchmark headline — RMSE at best R² (`presentation_benchmarks.ipynb` table)

**Brad (test Brad):** best on row shown — **XGB**, R² **0.572**, RMSE **3.106**, scenario **A**, variant **all-long**.

**Liberman (test Liberman):** **XGB**, R² **0.655**, RMSE **2.066**, scenario **C**, variant **even-wide**.

*(Full strip plot: `figures/presentation/benchmark_rmse.png`.)*

---

## Plot assets (optional PNGs)

Relative to `ABR2synapse/`:

| Asset | Path |
|-------|------|
| RMSE benchmark | `figures/presentation/benchmark_rmse.png` |
| R² benchmark | `figures/presentation/benchmark_r2.png` |
| Stage 1 (optional) | `figures/presentation/stage1_wide_rf.png` |
| Feature explainer PNGs | `figures/presentation/features/` — includes `feat_amplitude.png`, `feat_waveform_slope.png`, `feat_total_variance.png`, `feat_distance.png`, `feat_peak_curvature.png`, `feat_trough_curvature.png`, etc. |
