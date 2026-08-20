#!/usr/bin/env bash
# Orchestrates the whole distillation run inside the container.
#
# Every stage is idempotent: it writes $OUT_ROOT/<stage>.done on success and is
# skipped if that marker already exists. A job killed halfway can be resubmitted
# unchanged and will pick up where it stopped instead of paying for the GPU
# seconds again.
set -euo pipefail

DISTILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export STAGES="${STAGES:-label,train,eval,export}"
export DATA_ROOT="${DATA_ROOT:-/data}"
export OUT_ROOT="${OUT_ROOT:-/out}"
export TEACHER="${TEACHER:-maskomaly_id}"
export ITERS="${ITERS:-10000}"
export CROP="${CROP:-768}"
export BATCH="${BATCH:-16}"
export TEACHER_WEIGHTS="${TEACHER_WEIGHTS:-${DATA_ROOT}/ckpt/model_final_17c1ee.pkl}"
export PYTHONUNBUFFERED=1

mkdir -p "$OUT_ROOT" "$DATA_ROOT"

# Data staging. DataSphere's S3 mounts require a connector that can only be
# created in the project UI, so the container fetches its own inputs from
# Object Storage instead. Skipped entirely when S3_BUCKET is unset, which is
# how the image runs locally against a bind mount.
stage_inputs() {
  [[ -z "${S3_BUCKET:-}" ]] && { log "S3_BUCKET unset, using ${DATA_ROOT} as-is"; return 0; }
  if [[ -d "${DATA_ROOT}/frames" ]]; then
    log "inputs already staged, skipping download"
    return 0
  fi
  log "downloading inputs from s3://${S3_BUCKET}"
  python3 "${DISTILL_DIR}/s3sync.py" get data/raas-data.tar.gz /tmp/raas-data.tar.gz
  # The archive carries a raas-data/ top level; strip it so the layout matches
  # what common.list_corpus expects under DATA_ROOT.
  tar xzf /tmp/raas-data.tar.gz -C "$DATA_ROOT" --strip-components=1
  rm -f /tmp/raas-data.tar.gz
  python3 "${DISTILL_DIR}/s3sync.py" get data/ckpt/model_final_17c1ee.pkl \
      "${DATA_ROOT}/ckpt/model_final_17c1ee.pkl"
  log "inputs staged: $(find "${DATA_ROOT}/frames" -type f | wc -l) frames"
}

publish_outputs() {
  [[ -z "${S3_BUCKET:-}" ]] && return 0
  log "publishing results to s3://${S3_BUCKET}/out"
  for directory in checkpoints eval export; do
    python3 "${DISTILL_DIR}/s3sync.py" put-dir "${OUT_ROOT}/${directory}" "out/${directory}" || true
  done
  python3 "${DISTILL_DIR}/s3sync.py" put "${OUT_ROOT}/summary.json" out/summary.json || true
}
trap publish_outputs EXIT

log() { printf '\n=== %s | %s\n' "$(date -u +%H:%M:%S)" "$*"; }

stage_enabled() {
  [[ ",${STAGES}," == *",$1,"* ]]
}

run_stage() {
  local name="$1"; shift
  local marker="${OUT_ROOT}/${name}.done"

  if ! stage_enabled "$name"; then
    log "stage ${name}: not in STAGES=${STAGES}, skipping"
    return 0
  fi
  if [[ -f "$marker" ]]; then
    log "stage ${name}: already complete (${marker}), skipping"
    return 0
  fi

  log "stage ${name}: starting"
  local started=$SECONDS
  python3 "$@"
  printf '%s\n' "$(date -u +%FT%TZ)" > "$marker"
  log "stage ${name}: done in $((SECONDS - started))s"
}

log "RAAS distillation
  STAGES=${STAGES}  TEACHER=${TEACHER}
  DATA_ROOT=${DATA_ROOT}  OUT_ROOT=${OUT_ROOT}
  ITERS=${ITERS}  CROP=${CROP}  BATCH=${BATCH}"

python3 - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY

stage_inputs

run_stage label  "${DISTILL_DIR}/stages/label.py"
run_stage train  "${DISTILL_DIR}/stages/train.py"
run_stage eval   "${DISTILL_DIR}/stages/evaluate.py"
run_stage export "${DISTILL_DIR}/stages/export.py"

log "collecting summary"
python3 - <<'PY'
import json, os, pathlib
out = pathlib.Path(os.environ["OUT_ROOT"])
summary = {}
for key, relative in (
    ("label", "targets/label_stats.json"),
    ("train", "checkpoints/train_stats.json"),
    ("eval", "eval/student_metrics.json"),
    ("export", "export/export_stats.json"),
):
    path = out / relative
    if path.is_file():
        summary[key] = json.loads(path.read_text())
(out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

log "all done -> ${OUT_ROOT}/summary.json"
