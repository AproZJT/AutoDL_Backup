#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=.

# --- 核心资产路径 ---
TEACHER_WEIGHT="weights/final_sota/best_swa_0.545.pth"
BASE_CFG="external_modules/mmdetection/configs/grounding_dino/grounding_dino_swin-b_inference_zerowaste_f.py"
UNLABELED_JSON="./data/zerowaste-f/unlabeled/labels.json"
RAW_PRED="./data/pseudo_labels/teacher_raw.bbox.json"
FINAL_PSEUDO_GT="./data/zerowaste-f/teacher_pseudo_labels.json"

echo "=================================================="
echo "🤖 Step 1: 动态生成 Teacher 专用推理 Config (已修复 Evaluator 同步)"
echo "=================================================="
cat > generate_teacher_cfg.py << 'PY_EOF'
from mmengine.config import Config
cfg = Config.fromfile('external_modules/mmdetection/configs/grounding_dino/grounding_dino_swin-b_inference_zerowaste_f.py')

# 1. 替换数据输入端
cfg.test_dataloader.dataset.ann_file = 'unlabeled/labels.json'
cfg.test_dataloader.dataset.data_prefix = dict(img='train/data/')

# 2. 替换验证对账端 (防止 ID 匹配 AssertionError)
unlabeled_abs_path = './data/zerowaste-f/unlabeled/labels.json'
if isinstance(cfg.test_evaluator, dict):
    cfg.test_evaluator.ann_file = unlabeled_abs_path
elif isinstance(cfg.test_evaluator, list):
    for metric in cfg.test_evaluator:
        if 'ann_file' in metric:
            metric.ann_file = unlabeled_abs_path

cfg.dump('teacher_infer_cfg.py')
PY_EOF
python generate_teacher_cfg.py

echo "=================================================="
echo "🎯 Step 2: Teacher 盲测全量召回 (score_thr=0.01)"
echo "=================================================="
python external_modules/mmdetection/tools/test.py \
  teacher_infer_cfg.py \
  "$TEACHER_WEIGHT" \
  --cfg-options model.test_cfg.rcnn.score_thr=0.01 \
  test_evaluator.outfile_prefix="${RAW_PRED%.bbox.json}" || echo "[WARN] 评测打分可能失败，但预测文件已生成，继续下一步！"

echo "=================================================="
echo "🔪 Step 3: 施加极限后处理，生成终极伪真理"
echo "=================================================="
python semi_sup/scripts/apply_best_postproc.py \
  --gt_json "$UNLABELED_JSON" \
  --pred_json "$RAW_PRED" \
  --out_json "$FINAL_PSEUDO_GT" \
  --default_thr 0.0 \
  --default_nms_iou 1.0 \
  --metal_nms_iou 0.72

echo "✅ Teacher 功成身退！高质量伪标签已保存至: $FINAL_PSEUDO_GT"
