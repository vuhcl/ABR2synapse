#!/usr/bin/env bash
# Create stacked split/27+ branches from dirty Vu working tree (single commit each).
set -euo pipefail
cd "$(dirname "$0")/.."

BASE="${1:-Vu}"
BACKUP_REF="refs/backup/pre-split-$(date +%s)"
SHA=$(git stash create "pre-split-series-27" || true)
if [[ -n "${SHA:-}" ]]; then
  git update-ref "$BACKUP_REF" "$SHA"
  echo "Backup ref: $BACKUP_REF -> $SHA"
fi

git checkout "$BASE"

# name|space-separated files
slices=(
  "split/27-stage1-wide|utils/stage1_wide_report.py utils/benchmark_metrics.py scripts/run_stage1_export.py scripts/create_abr_stage1_wide_nb.py scripts/smoke_stage1_heatmap.py scripts/smoke_stage1_pooled.py abr_stage1_wide.ipynb"
  "split/28-stage2-sklearn|utils/stage2_sklearn.py utils/stage2_synthesis_plot.py utils/stage2_export.py"
  "split/29-liberman-true-noise|utils/liberman_classical.py scripts/smoke_liberman_classical.py scripts/rebuild_true_noise_oof.py scripts/create_liberman_synapse_comparison_true_noise_nb.py"
  "split/30-stage2-hp|utils/stage2_hp.py scripts/run_stage2_hp_ab.py scripts/run_stage2_hp_true_noise.py scripts/create_stage2_hp_tuning_true_noise_nb.py"
  "split/31-stage2-synthesis-cv|utils/stage2_synthesis_cv.py scripts/smoke_synthesis_stage1.py scripts/smoke_synthesis_cv_true_noise.py scripts/run_synthesis_cv_refresh.py scripts/create_stage2_synthesis_cv_true_noise_nb.py"
  "split/32-stage2-xgb-shap|utils/stage2_xgb_shap.py"
  "split/33-stage2-mlp-shap|utils/stage2_mlp_shap.py utils/paper_shap_figures.py scripts/run_mlp_shap_cohort.py scripts/smoke_mlp_shap.py scripts/verify_mlp_shap_exports.py scripts/export_paper_shap_figures.py scripts/smoke_paper_shap_figures.py scripts/create_mlp_shap_nb.py"
  "split/34-stage1-rf-shap|utils/stage1_rf_shap.py scripts/create_stage1_rf_shap_nb.py scripts/smoke_stage1_rf_shap.py abr_stage1_rf_shap.ipynb"
  "split/35-univariate-eda|utils/abr_univariate_eda.py scripts/smoke_abr_univariate_eda.py abr_univariate_eda.ipynb figures/eda/figure_04_caption.txt figures/eda/figure_05_caption.txt figures/eda/figure_06_caption.txt figures/eda/figure_07_caption.txt figures/eda/figure_08_caption.txt figures/eda/figure_combined_caption.txt"
  "split/36-nn-colab|utils/nn_colab_export.py utils/nn_colab_train.py utils/nn_stage2.py utils/nn_stage2_data.py scripts/create_nn_colab_nb.py scripts/export_liberman_t5_hp.py scripts/merge_liberman_hp.py scripts/run_liberman_nn_hp_comparison.py"
  "split/37-deck-storyline|utils/deck_storyline.py presentation_deck.ipynb"
  "split/38-comparison-nbs|abr_liberman_synapse_comparison.ipynb abr_liberman_synapse_comparison_true_noise.ipynb scripts/create_liberman_synapse_comparison_nb.py"
  "split/39-stage2-nbs|abr_stage2_hp_tuning.ipynb abr_stage2_hp_tuning_true_noise.ipynb abr_stage2_synthesis_cv.ipynb abr_stage2_synthesis_cv_true_noise.ipynb abr_nn_stage2_colab.ipynb"
  "split/40-shap-nbs|abr_stage2_mlp_shap.ipynb abr_stage2_xgb_shap.ipynb"
  "split/41-misc-nbs|models.ipynb scripts/smoke_paper_cleanup.py"
  "split/42-split-pr-scripts|scripts/create_split_prs.sh scripts/create_split_prs_api.sh scripts/run_split_series_27.sh"
)

titles=(
  "feat(utils): extend stage-1 wide report and benchmark metrics"
  "feat(utils): stage-2 sklearn runners and synthesis plot helpers"
  "feat(utils): Liberman classical true-noise panel and OOF rebuild"
  "feat(utils): stage-2 HP tuning with true-noise scenarios"
  "feat(utils): synthesis CV orchestration and true-noise smokes"
  "feat(utils): add stage-2 XGB SHAP module"
  "feat(utils): extend MLP SHAP and paper figure exports"
  "feat(utils): add stage-1 RF SHAP module and notebook"
  "feat(utils): univariate EDA helpers and figure captions"
  "feat(utils): NN Colab export, train, and stage-2 data updates"
  "feat(utils): deck storyline and presentation deck updates"
  "notebooks: Liberman synapse comparison and true-noise variant"
  "notebooks: stage-2 HP tuning, synthesis CV, and Colab notebooks"
  "notebooks: update MLP and XGB SHAP analysis notebooks"
  "notebooks: models exploration and paper-cleanup smoke"
  "chore(scripts): split PR helpers for series 27+"
)

prev="$BASE"
for i in "${!slices[@]}"; do
  entry="${slices[$i]}"
  branch="${entry%%|*}"
  files="${entry#*|}"
  title="${titles[$i]}"
  n=$((i + 27))
  echo "=== ${n}: ${branch} (${title}) ==="
  git checkout -B "$branch" "$prev"
  # ponytail: explicit file list only — no git add .
  for f in $files; do
    if [[ -e "$f" ]]; then
      git add "$f"
    else
      echo "  skip missing: $f" >&2
    fi
  done
  if git diff --cached --quiet; then
    echo "  nothing staged; skipping commit" >&2
  else
    git commit -m "$(cat <<EOF
${title}

PR ${n} in the Vu split series (27+). Single-commit review slice.
EOF
)"
  fi
  prev="$branch"
done

git checkout "$BASE"
echo "Done. Tip: bash scripts/create_split_prs_series_27.sh"
