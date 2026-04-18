#!/usr/bin/env bash
set -eu

ROOT_DIR="$PWD"
GT_JSON="$ROOT_DIR/data/zerowaste-f/test/labels.json"
RAW_JSON="$ROOT_DIR/data/pseudo_labels/raw_s0010.bbox.json"
POST_SCRIPT="$ROOT_DIR/semi_sup/scripts/apply_best_postproc.py"
OUT_DIR="$ROOT_DIR/data/pseudo_labels/stage4_cpu_sweep"

mkdir -p "$OUT_DIR"

echo "=================================================="
echo "🎯 Stage 4: pure CPU post-process sweep"
echo "Base raw json: $RAW_JSON"
echo "Fixed: metal_nms_iou=0.72, default_nms_iou=1.0"
echo "GT: $GT_JSON"
echo "=================================================="

if [ ! -f "$RAW_JSON" ]; then
  echo "❌ ERROR: raw prediction file not found: $RAW_JSON"
  exit 1
fi

# --------------------------------------------------
# Phase 4-1: sweep default_thr
# --------------------------------------------------
echo "=================================================="
echo "🔍 Phase 4-1: sweep default_thr"
echo "=================================================="

for DTHR in 0.000 0.001 0.002 0.003 0.004; do
  TAG="default_${DTHR//./}"
  OUT_JSON="$OUT_DIR/post_${TAG}.json"
  
  echo "--------------------------------------------------"
  echo "⚙️ Running default_thr = $DTHR"
  python "$POST_SCRIPT" \
    --gt_json "$GT_JSON" \
    --pred_json "$RAW_JSON" \
    --out_json "$OUT_JSON" \
    --default_thr "$DTHR" \
    --default_nms_iou 1.0 \
    --metal_nms_iou 0.72 \
    --max_dets 100,300,1000
done

echo "=================================================="
echo "⏸️ Phase 4-1 完毕！安全暂停！"
echo "=================================================="
