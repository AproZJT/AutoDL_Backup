#!/usr/bin/env bash
set -eu

ROOT_DIR="$PWD"
GT_JSON="$ROOT_DIR/data/zerowaste-f/test/labels.json"
RAW_JSON="$ROOT_DIR/data/pseudo_labels/raw_s0010.bbox.json"
POST_SCRIPT="$ROOT_DIR/semi_sup/scripts/apply_best_postproc.py"
OUT_DIR="$ROOT_DIR/data/pseudo_labels/stage4_cpu_sweep"

mkdir -p "$OUT_DIR"

echo "=================================================="
echo "🔍 Phase 4-3: sweep cardboard_thr (Above base score 0.010)"
echo "Fixed: default_thr=0.010, metal_nms_iou=0.712"
echo "=================================================="

# 基线是 0.010，从 0.012 开始扫
for CTHR in 0.012 0.015 0.018 0.022; do
  TAG="card_${CTHR//./}_d0010"
  OUT_JSON="$OUT_DIR/post_${TAG}.json"
  
  echo "--------------------------------------------------"
  echo "⚙️ Running cardboard_thr = $CTHR"
  python "$POST_SCRIPT" \
    --gt_json "$GT_JSON" \
    --pred_json "$RAW_JSON" \
    --out_json "$OUT_JSON" \
    --default_thr 0.010 \
    --cardboard_thr "$CTHR" \
    --default_nms_iou 1.0 \
    --metal_nms_iou 0.712 \
    --max_dets 100,300,1000
done

echo "=================================================="
echo "⏸️ Phase 4-3 完毕！Teacher 时代的最后宣判！"
echo "=================================================="
