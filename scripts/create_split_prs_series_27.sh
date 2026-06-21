#!/usr/bin/env bash
# Open stacked PRs for split/27+ series (base Vu, then each prior split branch).
set -euo pipefail
cd "$(dirname "$0")/.."

if ! gh auth status >/dev/null 2>&1; then
  echo "gh is not authenticated. After pushing branches, run:" >&2
  echo "  gh auth login" >&2
  echo "  bash scripts/create_split_prs_series_27.sh" >&2
  exit 1
fi

prs=(
  "Vu|split/27-stage1-wide|feat(utils): extend stage-1 wide report and benchmark metrics"
  "split/27-stage1-wide|split/28-stage2-sklearn|feat(utils): stage-2 sklearn runners and synthesis plot helpers"
  "split/28-stage2-sklearn|split/29-liberman-true-noise|feat(utils): Liberman classical true-noise panel and OOF rebuild"
  "split/29-liberman-true-noise|split/30-stage2-hp|feat(utils): stage-2 HP tuning with true-noise scenarios"
  "split/30-stage2-hp|split/31-stage2-synthesis-cv|feat(utils): synthesis CV orchestration and true-noise smokes"
  "split/31-stage2-synthesis-cv|split/32-stage2-xgb-shap|feat(utils): add stage-2 XGB SHAP module"
  "split/32-stage2-xgb-shap|split/33-stage2-mlp-shap|feat(utils): extend MLP SHAP and paper figure exports"
  "split/33-stage2-mlp-shap|split/34-stage1-rf-shap|feat(utils): add stage-1 RF SHAP module and notebook"
  "split/34-stage1-rf-shap|split/35-univariate-eda|feat(utils): univariate EDA helpers and figure captions"
  "split/35-univariate-eda|split/36-nn-colab|feat(utils): NN Colab export, train, and stage-2 data updates"
  "split/36-nn-colab|split/37-deck-storyline|feat(utils): deck storyline and presentation deck updates"
  "split/37-deck-storyline|split/38-comparison-nbs|notebooks: Liberman synapse comparison and true-noise variant"
  "split/38-comparison-nbs|split/39-stage2-nbs|notebooks: stage-2 HP tuning, synthesis CV, and Colab notebooks"
  "split/39-stage2-nbs|split/40-shap-nbs|notebooks: update MLP and XGB SHAP analysis notebooks"
  "split/40-shap-nbs|split/41-misc-nbs|notebooks: models exploration and paper-cleanup smoke"
  "split/41-misc-nbs|split/42-split-pr-scripts|chore(scripts): split PR helpers for series 27+"
)

n=26
for entry in "${prs[@]}"; do
  IFS='|' read -r base head title <<< "$entry"
  n=$((n + 1))
  echo "=== PR ${n}: ${title} ==="
  gh pr create --base "$base" --head "$head" --title "$title" --body "PR ${n} in the Vu split series (27+). Single-commit review slice."
done

echo "Done. Open PRs: gh pr list --head split/"
