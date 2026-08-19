#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 DATASETS_ROOT OUTPUT_ROOT SAM_CHECKPOINT" >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RAAS_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
DATASETS_ROOT=$(realpath "$1")
OUTPUT_ROOT=$(realpath -m "$2")
SAM_CHECKPOINT=$(realpath "$3")
RAAS_ENV=${RAAS_ENV:-raas}
OBJECTOMALY_ENV=${OBJECTOMALY_ENV:-objectomaly}
CONFIG="$RAAS_ROOT/Maskomaly/configs/objectomaly_global_fusion.json"
CACHE_ROOT="$OUTPUT_ROOT/cache"
REFINED_ROOT="$OUTPUT_ROOT/refined"
METRICS_ROOT="$OUTPUT_ROOT/metrics"
MODELS=(maskomaly maskomaly_id maskomaly_ood)

mkdir -p "$CACHE_ROOT" "$REFINED_ROOT" "$METRICS_ROOT"

echo "[1/4] RAAS inference and semantic maps"
conda run --no-capture-output -n "$RAAS_ENV" \
  python "$RAAS_ROOT/Maskomaly/scripts/export_objectomaly_inputs.py" \
  --datasets-root "$DATASETS_ROOT" \
  --output "$CACHE_ROOT" \
  --models "${MODELS[@]}"

echo "[2/4] Shared SAM mask cache and first refinement"
conda run --no-capture-output -n "$OBJECTOMALY_ENV" \
  python "$RAAS_ROOT/Maskomaly/scripts/run_objectomaly_refinement.py" \
  --manifest "$CACHE_ROOT/manifest-maskomaly.json" \
  --output "$REFINED_ROOT" \
  --config "$CONFIG" \
  --sam-checkpoint "$SAM_CHECKPOINT" \
  --phase all

echo "[3/4] Remaining OASC + global CLIP fusion + MBP refinements"
for model in maskomaly_id maskomaly_ood; do
  conda run --no-capture-output -n "$OBJECTOMALY_ENV" \
    python "$RAAS_ROOT/Maskomaly/scripts/run_objectomaly_refinement.py" \
    --manifest "$CACHE_ROOT/manifest-${model}.json" \
    --output "$REFINED_ROOT" \
    --config "$CONFIG" \
    --phase refine
done

echo "[4/4] Official SMIYC metrics"
for model in "${MODELS[@]}"; do
  conda run --no-capture-output -n "$RAAS_ENV" \
    python "$RAAS_ROOT/Maskomaly/scripts/import_objectomaly_outputs.py" \
    --manifest "$REFINED_ROOT/manifest-objectomaly-${model}.json" \
    --datasets-root "$DATASETS_ROOT" \
    --output "$METRICS_ROOT" \
    --method-name "objectomaly_global_${model}"
done

echo "Metrics: $METRICS_ROOT/summary.csv"
echo "Timings: $REFINED_ROOT/timings-summary-<model>.json"
