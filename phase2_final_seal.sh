#!/usr/bin/env bash
set -eu

ROOT_DIR="$PWD"
POST_SCRIPT="$ROOT_DIR/semi_sup/scripts/apply_best_postproc.py"

# ✅ 已经替换为你真实的无标签 JSON 路径
UNLABELED_JSON="$ROOT_DIR/data/zerowaste-f/unlabeled/labels.json"
RAW_PRED="$ROOT_DIR/data/pseudo_labels/unlabeled_raw.bbox.json"
FINAL_PSEUDO_GT="$ROOT_DIR/data/zerowaste-f/final_pseudo_labels.json"

echo "=================================================="
echo "🛡️ 封板行动：生成 Student 训练用最终伪标签"
echo "=================================================="

if [ ! -f "$UNLABELED_JSON" ]; then
    echo "❌ 依然找不到无标签壳文件: $UNLABELED_JSON，请检查路径！"
    exit 1
fi

python "$POST_SCRIPT" \
  --gt_json "$UNLABELED_JSON" \
  --pred_json "$RAW_PRED" \
  --out_json "$FINAL_PSEUDO_GT" \
  --default_thr 0.010 \
  --default_nms_iou 1.0 \
  --metal_nms_iou 0.712 \
  --max_dets 1000

echo "✅ 成功！最终伪标签已落盘：$FINAL_PSEUDO_GT"
