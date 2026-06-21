#!/usr/bin/env bash
# Create stacked PRs via GitHub REST API — no `gh auth login` required.
#
# Usage (Terminal.app):
#   export GH_TOKEN="ghp_..."   # classic token with repo scope
#   bash scripts/create_split_prs_api.sh          # PRs 4-26 (1-3 merged manually)
#   bash scripts/create_split_prs_api.sh 4        # same, explicit start
#   bash scripts/create_split_prs_api.sh 10        # resume from PR 10
#
# Get token: https://github.com/settings/tokens/new?scopes=repo
set -euo pipefail

REPO="vuhcl/ABR2synapse"
API="https://api.github.com/repos/${REPO}/pulls"

if [[ -z "${GH_TOKEN:-}" && -z "${GITHUB_TOKEN:-}" ]]; then
  echo "Set GH_TOKEN first (do NOT run gh auth login):" >&2
  echo '  export GH_TOKEN="ghp_..."' >&2
  exit 1
fi
TOKEN="${GH_TOKEN:-${GITHUB_TOKEN}}"

# Full catalog (PR number = array index + 1)
all_prs=(
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

START=${1:-4}
if [[ "$START" -lt 1 || "$START" -gt ${#all_prs[@]} ]]; then
  echo "Start PR must be 1-${#all_prs[@]}, got: $START" >&2
  exit 1
fi

for ((i = START - 1; i < ${#all_prs[@]}; i++)); do
  entry="${all_prs[$i]}"
  IFS='|' read -r base head title <<< "$entry"
  n=$((i + 1))
  body="PR ${n}/26 in the Vu split series. Single-commit review slice."
  payload=$(python3 -c 'import json,sys; print(json.dumps({"title":sys.argv[1],"head":sys.argv[2],"base":sys.argv[3],"body":sys.argv[4]}))' \
    "$title" "$head" "$base" "$body")

  echo "=== PR ${n}/26: ${title} ==="
  resp=$(curl -sS --http1.1 -w "\n%{http_code}" -X POST \
    -H "Authorization: Bearer ${TOKEN}" \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "$API" \
    -d "$payload")
  http_code=$(echo "$resp" | tail -1)
  body_json=$(echo "$resp" | sed '$d')

  if [[ "$http_code" == "201" ]]; then
    url=$(echo "$body_json" | python3 -c 'import json,sys; print(json.load(sys.stdin)["html_url"])')
    echo "  OK: $url"
  elif echo "$body_json" | grep -q '"message": "Validation Failed"' && echo "$body_json" | grep -q 'already exists'; then
    echo "  SKIP: PR already exists for ${head} -> ${base}"
  else
    echo "  FAIL (HTTP ${http_code}): ${body_json}" >&2
    exit 1
  fi
done

echo "Done."
