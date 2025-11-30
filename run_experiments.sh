#!/usr/bin/env bash
# Run RAG experiments (1..7) and save outputs to `output/experiment_X.jsonl`.
# Usage: ./run_experiments.sh <N|all>  (e.g. ./run_experiments.sh 1) or ./run_experiments.sh all

set -uo pipefail

DATASET="dataset/filtered_QA_with_injections_10pct.jsonl"
mkdir -p output logs

# Configuration for experiments (index 1..7)
TARGET=("" "llama4" "llama4" "llama4" "gemini2.5-flash" "llama3.3" "llama4" "llama4")
VERIFIER=("" "gemini2.5-flash" "gemini2.5-flash" "gemini2.5-flash" "gemini2.5-flash" "gemini2.5-flash" "llama4" "llama3.3")
PROMPT_FORMAT=("" "query_before_context" "query_before_context" "context_before_query" "query_before_context" "query_before_context" "query_before_context" "query_before_context")
USE_SYSTEM=("" "false" "true" "false" "false" "false" "false" "false")
NUM_QUERIES=("" "500" "500" "500" "500" "500" "500" "500")

function usage() {
  echo "Usage: $0 <experiment-number|all>"
  echo "Examples:"
  echo "  $0 1         # run experiment 1"
  echo "  $0 all       # run all experiments 1..7"
  exit 1
}

if [ $# -lt 1 ]; then
  usage
fi

SELECTION=("$@")
if [ "${SELECTION[0]}" = "all" ]; then
  SELECTION=(1 2 3 4 5 6 7)
fi

SUCCESS=()
FAILURE=()

for s in "${SELECTION[@]}"; do
  if ! [[ "$s" =~ ^[0-9]+$ ]]; then
    echo "Invalid selection: $s" >&2
    continue
  fi
  if [ "$s" -lt 1 ] || [ "$s" -gt 7 ]; then
    echo "Experiment number must be 1..7 (got: $s)" >&2
    continue
  fi

  target=${TARGET[$s]}
  verifier=${VERIFIER[$s]}
  prompt_format=${PROMPT_FORMAT[$s]}
  use_system=${USE_SYSTEM[$s]}
  num_queries=${NUM_QUERIES[$s]}

  outfile="output/experiment_${s}.jsonl"

  echo "--- Running experiment $s ---"
  echo " target:   $target"
  echo " verifier: $verifier"
  echo " prompt:   $prompt_format"
  echo " system:   $use_system"
  echo " queries:  $num_queries"
  echo " output:   $outfile"

  cmd=(uv run chatbot/main.py --log-level debug --target-model "$target" --verifier-model "$verifier" --prompt-format "$prompt_format")
  if [ "$use_system" = "false" ]; then
    cmd+=(--no-system-prompt)
  fi
  cmd+=(--output-file "$outfile" --num-queries "$num_queries" --prompt-dataset-file "$DATASET")

  echo "Running: ${cmd[*]}"
  if "${cmd[@]}"; then
    echo "Experiment $s completed successfully." | tee -a logs/experiments.log
    SUCCESS+=("$s")
  else
    echo "Experiment $s FAILED." | tee -a logs/experiments.log
    FAILURE+=("$s")
  fi
  echo
done

echo "=== Summary ==="
echo "Succeeded: ${SUCCESS[*]:-none}"
echo "Failed:    ${FAILURE[*]:-none}"

if [ ${#FAILURE[@]} -gt 0 ]; then
  exit 2
fi
