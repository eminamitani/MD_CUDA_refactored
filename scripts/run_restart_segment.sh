#!/usr/bin/env bash
set -euo pipefail

: "${MD_CUDA_BIN:?MD_CUDA_BIN is required}"
: "${MD_CONFIG:?MD_CONFIG is required}"
: "${CHECKPOINT_DIR:?CHECKPOINT_DIR is required}"

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${RUN_ROOT:-$(pwd)}"
MODEL_PATH="${MODEL_PATH:-}"
COMPLETE_MARKER="${COMPLETE_MARKER:-$RUN_ROOT/COMPLETE.json}"
CONTINUE_MARKER="${CONTINUE_MARKER:-$RUN_ROOT/CONTINUE_READY.json}"
MD_LOG="${MD_LOG:-$RUN_ROOT/md.segment.${PBS_JOBID:-${JOB_ID:-manual}}.log}"
EXPECTED_TARGET_STEP="${EXPECTED_TARGET_STEP:-}"

mkdir -p "$RUN_ROOT" "$CHECKPOINT_DIR"

validate_complete() {
  python3 -m json.tool "$COMPLETE_MARKER" >/dev/null
  if [[ -s "$CHECKPOINT_DIR/segment_manifest.json" ]]; then
    python3 "$SCRIPT_ROOT/validate_segmented_trajectory.py" \
      "$CHECKPOINT_DIR/segment_manifest.json" \
      --output "$RUN_ROOT/segment_validation.json"
    if [[ -n "$EXPECTED_TARGET_STEP" ]]; then
      python3 - "$CHECKPOINT_DIR/segment_manifest.json" "$EXPECTED_TARGET_STEP" <<'PY'
import json
import sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text())
expected = int(sys.argv[2])
segments = manifest.get("segments", [])
if not segments or max(int(item["end_step"]) for item in segments) != expected:
    raise SystemExit(f"segment manifest has not reached target step {expected}")
PY
    fi
  fi
}

if [[ -s "$COMPLETE_MARKER" ]]; then
  validate_complete
  echo "already complete and revalidated: $COMPLETE_MARKER"
  exit 0
fi

set +e
"$MD_CUDA_BIN" "$MD_CONFIG" >"$MD_LOG" 2>&1
md_rc=$?
set -e

validator=(
  python3 "$SCRIPT_ROOT/validate_md_checkpoint.py"
  "$CHECKPOINT_DIR"
  --config "$MD_CONFIG"
  --output "$RUN_ROOT/checkpoint_validation.json"
)
if [[ -n "$MODEL_PATH" ]]; then
  validator+=(--model "$MODEL_PATH")
fi

if [[ "$md_rc" -eq 75 ]]; then
  "${validator[@]}" --require
  python3 - "$CONTINUE_MARKER" "$MD_LOG" "$md_rc" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path, log, returncode = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
path.write_text(json.dumps({
    "status": "continue_ready",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "md_log": log,
    "md_returncode": returncode,
}, indent=2) + "\n")
PY
  echo "CONTINUE_READY"
  exit 0
fi

if [[ "$md_rc" -ne 0 ]]; then
  echo "MD_CUDA failed with return code $md_rc" >&2
  exit "$md_rc"
fi

if [[ -s "$CHECKPOINT_DIR/segment_manifest.json" ]]; then
  python3 "$SCRIPT_ROOT/validate_segmented_trajectory.py" \
    "$CHECKPOINT_DIR/segment_manifest.json" \
    --output "$RUN_ROOT/segment_validation.json"
fi
if [[ -n "$EXPECTED_TARGET_STEP" ]]; then
  python3 - "$CHECKPOINT_DIR/segment_manifest.json" "$EXPECTED_TARGET_STEP" <<'PY'
import json
import sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text())
expected = int(sys.argv[2])
segments = manifest.get("segments", [])
if not segments or max(int(item["end_step"]) for item in segments) != expected:
    raise SystemExit(f"completed MD did not reach target step {expected}")
PY
fi
python3 - "$COMPLETE_MARKER" "$MD_LOG" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path, log = Path(sys.argv[1]), sys.argv[2]
path.write_text(json.dumps({
    "status": "complete",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "md_log": log,
    "md_returncode": 0,
}, indent=2) + "\n")
PY
rm -f "$CONTINUE_MARKER"
validate_complete
echo "COMPLETE"
