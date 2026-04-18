#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTHONPATH=.

CONFIG="external_modules/mmdetection/configs/grounding_dino/grounding_dino_swin-b_inference_zerowaste_f.py"
GT_JSON="./data/zerowaste-f/test/labels.json"

# 替换为你真实存在的“精英权重”
CKPT_1="weights/final_sota/best_swa_0.545.pth"
CKPT_2="weights/gdino-swin-b/zerowaste_SWA_fused_ultimate.pth"
CKPT_3="weights/gdino-swin-b/zerowaste_semi-sup_best_coco_bbox_mAP.pth"

echo "===== 1. 检查文件是否存在 ====="
for f in "$CONFIG" "$GT_JSON" "$CKPT_1" "$CKPT_2" "$CKPT_3"; do
  [ -f "$f" ] || { echo "[ERROR] missing file: $f"; exit 1; }
done
echo "✅ 所有精英权重均已就位！"

mkdir -p data/pseudo_labels

echo "===== 2. 导出三路精英模型的底层预测 (低阈值) ====="
python external_modules/mmdetection/tools/test.py "$CONFIG" "$CKPT_1" \
  --cfg-options model.test_cfg.rcnn.score_thr=0.01 test_evaluator.outfile_prefix=./data/pseudo_labels/test_elite1

python external_modules/mmdetection/tools/test.py "$CONFIG" "$CKPT_2" \
  --cfg-options model.test_cfg.rcnn.score_thr=0.01 test_evaluator.outfile_prefix=./data/pseudo_labels/test_elite2

python external_modules/mmdetection/tools/test.py "$CONFIG" "$CKPT_3" \
  --cfg-options model.test_cfg.rcnn.score_thr=0.01 test_evaluator.outfile_prefix=./data/pseudo_labels/test_elite3

echo "===== 3. 套用 0.546755 的最强后处理清洗三路数据 ====="
python semi_sup/scripts/apply_best_postproc.py \
  --gt_json "$GT_JSON" --pred_json ./data/pseudo_labels/test_elite1.bbox.json \
  --out_json ./data/pseudo_labels/test_elite1_post.bbox.json \
  --default_thr 0.0 --default_nms_iou 1.0 --metal_nms_iou 0.72 --skip_eval

python semi_sup/scripts/apply_best_postproc.py \
  --gt_json "$GT_JSON" --pred_json ./data/pseudo_labels/test_elite2.bbox.json \
  --out_json ./data/pseudo_labels/test_elite2_post.bbox.json \
  --default_thr 0.0 --default_nms_iou 1.0 --metal_nms_iou 0.72 --skip_eval

python semi_sup/scripts/apply_best_postproc.py \
  --gt_json "$GT_JSON" --pred_json ./data/pseudo_labels/test_elite3.bbox.json \
  --out_json ./data/pseudo_labels/test_elite3_post.bbox.json \
  --default_thr 0.0 --default_nms_iou 1.0 --metal_nms_iou 0.72 --skip_eval

echo "===== 4. 启动最终的 WBF 网格搜索 ====="
for IOU in 0.55 0.60 0.65; do
  for SKIP in 0.001 0.01; do
    echo ">>> Elite WBF iou=$IOU skip=$SKIP"
    python semi_sup/scripts/offline_wbf_eval.py \
      --gt_json "$GT_JSON" \
      --pred_jsons ./data/pseudo_labels/test_elite1_post.bbox.json ./data/pseudo_labels/test_elite2_post.bbox.json ./data/pseudo_labels/test_elite3_post.bbox.json \
      --wbf_iou_thr $IOU \
      --skip_box_thr $SKIP
  done
done
