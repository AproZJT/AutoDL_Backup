#!/usr/bin/env bash
set -eu

ROOT_DIR="$PWD"
BEST_PTH="$ROOT_DIR/weights/final_sota/best_swa_0.545.pth"
TEST_SCRIPT="$ROOT_DIR/external_modules/mmdetection/tools/test.py"
CFG_FILE="$ROOT_DIR/external_modules/mmdetection/configs/grounding_dino/grounding_dino_swin-b_inference_zerowaste_f.py"
GT_JSON="$ROOT_DIR/data/zerowaste-f/test/labels.json"
RAW_DIR="$ROOT_DIR/data/pseudo_labels"
OUT_DIR="$ROOT_DIR/data/pseudo_labels/stage1_score_thr"

mkdir -p "$RAW_DIR"
mkdir -p "$OUT_DIR"

echo "=================================================="
echo "Stage 1: score_thr narrow sweep"
echo "Fixed postproc: metal_nms_iou=0.72, default_thr=0.0, default_nms_iou=1.0"
echo "GT for evaluation: $GT_JSON"
echo "=================================================="

for STHR in 0.008 0.010 0.012; do
  TAG="s${STHR//./}"
  RAW_PREFIX="$RAW_DIR/raw_${TAG}"
  RAW_JSON="${RAW_PREFIX}.bbox.json"
  OUT_JSON="$OUT_DIR/post_${TAG}.json"

  echo "=================================================="
  echo "🔥 Running score_thr = $STHR"
  echo "Raw output: $RAW_JSON"
  echo "Post output: $OUT_JSON"
  echo "=================================================="

  python "$TEST_SCRIPT" \
    "$CFG_FILE" \
    "$BEST_PTH" \
    --cfg-options model.test_cfg.rcnn.score_thr=$STHR \
    test_evaluator.outfile_prefix="$RAW_PREFIX"

  if [ ! -f "$RAW_JSON" ]; then
    echo "❌ ERROR: raw prediction file not found: $RAW_JSON"
    exit 1
  fi

  python "$ROOT_DIR/semi_sup/scripts/apply_best_postproc.py" \
    --gt_json "$GT_JSON" \
    --pred_json "$RAW_JSON" \
    --out_json "$OUT_JSON" \
    --default_thr 0.0 \
    --default_nms_iou 1.0 \
    --metal_nms_iou 0.72 \
    --max_dets 100,300,1000
done

echo "=================================================="
echo "✅ Stage 1 sweep finished."
echo "Check outputs under: $OUT_DIR"
echo "=================================================="
