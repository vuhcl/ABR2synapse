#!/usr/bin/env bash
# Create 26 stacked PRs for the Vu split series.
# Prerequisite: gh auth login (see README in script header below).
set -euo pipefail
cd "$(dirname "$0")/.."

if ! gh auth status >/dev/null 2>&1; then
  echo "gh is not authenticated. Run token login first:" >&2
  echo '  gh auth login --hostname github.com --git-protocol https --with-token <<< "ghp_..."' >&2
  exit 1
fi

prs=(
  "split-base|split/01-gitignore|chore: sync monorepo gitignore and ABR2synapse layout docs"
  "split/01-gitignore|split/02-data-loader-paths|fix(utils): anchor Liberman data paths to repo root"
  "split/02-data-loader-paths|split/03-subject-cv|feat(utils): add subject-level grouped CV helpers"
  "split/03-subject-cv|split/04-benchmark-metrics|feat(utils): add cache path helpers and synthesis plot utilities"
  "split/04-benchmark-metrics|split/05-nn-stage2-data|feat(utils): extend stage-1 wide SPL levels and train pools"
  "split/05-nn-stage2-data|split/06-nn-stage2|refactor(utils): animal-level stage-1 wide and long stage-2 NN"
  "split/06-nn-stage2|split/07-ols-long|feat(utils): extract long-format OLS evaluation helper"
  "split/07-ols-long|split/08-stage2-sklearn|feat(utils): add two-stage RF/XGB sklearn runners"
  "split/08-stage2-sklearn|split/09-liberman-classical|feat(utils): add Liberman classical comparison panel"
  "split/09-liberman-classical|split/10-nn-colab-train|feat(utils): add portable Colab MLP/CNN training module"
  "split/10-nn-colab-train|split/11-stage2-export|feat(utils): export stage-2 HP tuning parquet bundles"
  "split/11-stage2-export|split/12-colab-export|feat(utils): Liberman NN Colab export and HP merge scripts"
  "split/12-colab-export|split/13-stage2-hp|feat(utils): stage-2 hyperparameter search for scenarios A/B/C"
  "split/13-stage2-hp|split/14-stage2-mlp-shap|feat(utils): MLP stage-2 SHAP module"
  "split/14-stage2-mlp-shap|split/15-stage2-synthesis-cv|feat(utils): synthesis CV orchestration, plot helpers, and smokes"
  "split/15-stage2-synthesis-cv|split/16-presentation-utils|feat(utils): deck and stage-1 presentation plot helpers"
  "split/16-presentation-utils|split/17-stage1-notebook|notebooks: add stage-1 wide evaluation notebook and export script"
  "split/17-stage1-notebook|split/18-comparison-notebook|notebooks: update wide/long comparison for paper cleanup"
  "split/18-comparison-notebook|split/19-nn-stage2-notebook|notebooks: slim abr_nn_stage2 to long stage-2 only"
  "split/19-nn-stage2-notebook|split/20-benchmarks-notebook|notebooks: wire presentation_benchmarks to stage-1 cache"
  "split/20-benchmarks-notebook|split/21-liberman-notebook|notebooks: add Liberman synapse comparison panel"
  "split/21-liberman-notebook|split/22-stage2-notebooks|notebooks: add HP tuning, synthesis CV, and Colab stage-2 notebooks"
  "split/22-stage2-notebooks|split/23-shap-notebooks|notebooks: add MLP and XGB SHAP analysis notebooks"
  "split/23-shap-notebooks|split/24-deck-notebooks|notebooks: add presentation deck and Liberman baseline"
  "split/24-deck-notebooks|split/25-misc-notebooks|notebooks: minor updates to data exploration and original copies"
  "split/25-misc-notebooks|split/26-verify|test: add paper-cleanup verification orchestrator"
)

n=0
for entry in "${prs[@]}"; do
  IFS='|' read -r base head title <<< "$entry"
  n=$((n + 1))
  echo "=== PR ${n}/26: ${title} ==="
  gh pr create --base "$base" --head "$head" --title "$title" --body "PR ${n}/26 in the Vu split series. Single-commit review slice."
done

echo "Done. Open PRs: gh pr list --head split/"
