#!/usr/bin/env bash
set -euo pipefail

SPLITS_DIR="../Datasets/Split"

TUNING_DIR="tuning_results"
STACKING_DIR="stacking_results"

TUNING_SCRIPT="train_individual_models.py"
STACKING_SCRIPT="run_stacking.py"

N_TRIALS=50
N_SPLITS=5
RANDOM_STATE=42
META_MODEL="ridge"

MODELS=(
  "lightgbm"
  "catboost"
  "random_forest"
  "xgboost"
  "mlp"
)

SKIP_TUNING=false
SKIP_STACKING=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tuning-dir)
      TUNING_DIR="$2"
      shift 2
      ;;
    --stacking-dir)
      STACKING_DIR="$2"
      shift 2
      ;;
    --tuning-script)
      TUNING_SCRIPT="$2"
      shift 2
      ;;
    --stacking-script)
      STACKING_SCRIPT="$2"
      shift 2
      ;;
    --n-trials)
      N_TRIALS="$2"
      shift 2
      ;;
    --n-splits)
      N_SPLITS="$2"
      shift 2
      ;;
    --random-state)
      RANDOM_STATE="$2"
      shift 2
      ;;
    --meta-model)
      META_MODEL="$2"
      shift 2
      ;;
    --models)
      shift
      MODELS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        MODELS+=("$1")
        shift
      done
      ;;
    --skip-tuning)
      SKIP_TUNING=true
      shift
      ;;
    --skip-stacking)
      SKIP_STACKING=true
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

echo "============================================================"
echo "Pipeline configuration"
echo "============================================================"
echo "Splits dir:      $SPLITS_DIR"
echo "Tuning dir:      $TUNING_DIR"
echo "Stacking dir:    $STACKING_DIR"
echo "Tuning script:   $TUNING_SCRIPT"
echo "Stacking script: $STACKING_SCRIPT"
echo "Trials:          $N_TRIALS"
echo "CV folds:        $N_SPLITS"
echo "Random state:    $RANDOM_STATE"
echo "Meta-model:      $META_MODEL"
echo "Models:          ${MODELS[*]}"
echo "Skip tuning:     $SKIP_TUNING"
echo "Skip stacking:   $SKIP_STACKING"
echo "============================================================"

required_files=(
  "$TUNING_SCRIPT"
  "$STACKING_SCRIPT"
  "$SPLITS_DIR/X_train_raw.parquet"
  "$SPLITS_DIR/X_test_raw.parquet"
  "$SPLITS_DIR/X_train_scaled_ohe.parquet"
  "$SPLITS_DIR/X_test_scaled_ohe.parquet"
  "$SPLITS_DIR/y_train.parquet"
  "$SPLITS_DIR/y_test.parquet"
)

for file in "${required_files[@]}"; do
  if [[ ! -f "$file" ]]; then
    echo "Missing required file: $file"
    exit 1
  fi
done

mkdir -p "$TUNING_DIR"
mkdir -p "$STACKING_DIR"

if [[ "$SKIP_TUNING" == false ]]; then
  echo
  echo "============================================================"
  echo "Step 1/2 — Individual model tuning"
  echo "============================================================"

  python "$TUNING_SCRIPT" \
    --splits-dir "$SPLITS_DIR" \
    --output-dir "$TUNING_DIR" \
    --n-trials "$N_TRIALS" \
    --n-splits "$N_SPLITS" \
    --random-state "$RANDOM_STATE" \
    --models "${MODELS[@]}"
else
  echo
  echo "Skipping tuning."
fi

if [[ "$SKIP_STACKING" == false ]]; then
  echo
  echo "============================================================"
  echo "Step 2/2 — Stacking"
  echo "============================================================"

  python "$STACKING_SCRIPT" \
    --splits-dir "$SPLITS_DIR" \
    --tuning-dir "$TUNING_DIR" \
    --output-dir "$STACKING_DIR" \
    --n-splits "$N_SPLITS" \
    --random-state "$RANDOM_STATE" \
    --models "${MODELS[@]}" \
    --meta-model "$META_MODEL"
else
  echo
  echo "Skipping stacking."
fi

echo
echo "============================================================"
echo "Pipeline finished successfully"
echo "============================================================"
echo "Tuning results:   $TUNING_DIR"
echo "Stacking results: $STACKING_DIR"