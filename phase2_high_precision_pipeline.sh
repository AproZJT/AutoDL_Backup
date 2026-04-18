#!/usr/bin/env bash
set -euo pipefail

RAW_PRED="./data/pseudo_labels/teacher_raw.bbox.json"
HIGH_PREC_GT="./data/zerowaste-f/teacher_pseudo_labels_high_prec.json"

echo "=================================================="
echo "🧹 1. 启动高阈值清洗 (宁缺毋滥模式)"
echo "=================================================="
# 将 default_thr 从 0.0 直接拉升到 0.4，砍掉至少 80% 的噪声框
python semi_sup/scripts/apply_best_postproc.py \
  --gt_json ./data/zerowaste-f/unlabeled/labels.json \
  --pred_json "$RAW_PRED" \
  --out_json ./data/pseudo_labels/temp_high_prec.json \
  --default_thr 0.4 \
  --default_nms_iou 1.0 \
  --metal_nms_iou 0.72

echo "=================================================="
echo "📦 2. 缝合 COCO 训练集格式"
echo "=================================================="
python - << 'PY'
import json
import os

gt_path = './data/zerowaste-f/unlabeled/labels.json'
temp_path = './data/pseudo_labels/temp_high_prec.json'
out_path = './data/zerowaste-f/teacher_pseudo_labels_high_prec.json'

with open(gt_path, 'r', encoding='utf-8') as f:
    coco_dict = json.load(f)
with open(temp_path, 'r', encoding='utf-8') as f:
    pseudo_list = json.load(f)

print(f"   >> 清洗后剩余高质量伪标签框数量: {len(pseudo_list)} 个 (对比之前的 89 万个)")

annotations = []
for i, pred in enumerate(pseudo_list):
    w, h = pred['bbox'][2], pred['bbox'][3]
    ann = {
        'id': i + 1,
        'image_id': pred['image_id'],
        'category_id': pred['category_id'],
        'bbox': pred['bbox'],
        'area': w * h,
        'iscrowd': 0,
        'score': pred.get('score', 1.0)
    }
    annotations.append(ann)

coco_dict['annotations'] = annotations
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(coco_dict, f)
os.remove(temp_path)
PY

echo "=================================================="
echo "🧬 3. 动态修改 Student Config 指向新数据"
echo "=================================================="
# 直接用 sed 命令替换 Config 里的伪标签路径
sed -i 's/teacher_pseudo_labels.json/teacher_pseudo_labels_high_prec.json/g' phase2_student_finetune.py

echo "✅ 高精清洗完成！Config 已更新！"
