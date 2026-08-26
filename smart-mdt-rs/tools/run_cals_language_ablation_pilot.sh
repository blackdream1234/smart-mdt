#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
crate_dir="$(cd "${script_dir}/.." && pwd)"
output_dir="${crate_dir}/experiment_artifacts/cals_language_ablation_pilot"

if [[ -e "${output_dir}" ]]; then
    echo "refusing to overwrite existing pilot output: ${output_dir}" >&2
    exit 2
fi

methods="unary,horn,antihorn,square2cnf,affine,cals_unary,cals_horn,cals_antihorn,cals_square2cnf,cals_affine,cals_compact_explain_unary,cals_compact_explain_horn,cals_compact_explain_antihorn,cals_compact_explain_square2cnf,cals_compact_explain_affine,cals,cals_compact_explain"
datasets="tic-tac-toe,car_evaluation-bin,balance-scale-bin,anneal,HTRU_2-bin"

cd "${crate_dir}"
started="${SECONDS}"
cargo run --release -- benchmark \
    --data ../data \
    --datasets "${datasets}" \
    --runs 3 \
    --depths 5,7 \
    --methods "${methods}" \
    --seed 42 \
    --strict-data-checks \
    --output "${output_dir}"
elapsed="$((SECONDS - started))"
printf '%s\n' "${elapsed}" > "${output_dir}/pilot_wall_time_seconds.txt"
printf '%s\n' "${methods}" > "${output_dir}/methods.txt"
printf '%s\n' "${datasets}" > "${output_dir}/datasets.txt"

python3 tools/cals_language_ablation_analysis.py \
    --input "${output_dir}/full_results.csv" \
    --output-dir "${output_dir}/analysis" \
    --bootstrap-resamples 10000 \
    --seed 42
