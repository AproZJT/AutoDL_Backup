#!/usr/bin/env bash
set -eu

ROOT_DIR="$PWD"

RAW_PRED="$ROOT_DIR/release_v1_0547/results/final_sota_raw.bbox.json"
FINAL_OUT="$ROOT_DIR/release_v1_0547/results/reproduced_0547.bbox.json"
POST_SCRIPT="$ROOT_DIR/semi_sup/scripts/apply_best_postproc.py"
GT_JSON="$ROOT_DIR/data/zerowaste-f/unlabeled/labels.json"

python "$POST_SCRIPT" \
  --gt_json "$GT_JSON" \
  --pred_json "$RAW_PRED" \
  --out_json "$FINAL_OUT" \
  --default_thr 0.0 \
  --metal_thr 0.0 \
  --default_nms_iou 1.0 \
  --metal_nms_iou 0.71 \
  --max_dets 500

echo "✅ Reproduction finished: $FINAL_OUT"
