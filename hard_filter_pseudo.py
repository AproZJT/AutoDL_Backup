import json

raw_path = './data/pseudo_labels/teacher_raw.bbox.json'
gt_path = './data/zerowaste-f/unlabeled/labels.json'
out_path = './data/zerowaste-f/teacher_pseudo_labels_high_prec.json'

print("🔄 正在加载原始 89 万个预测框...")
with open(gt_path, 'r', encoding='utf-8') as f:
    coco_dict = json.load(f)
with open(raw_path, 'r', encoding='utf-8') as f:
    raw_preds = json.load(f)

# --- 硬核物理过滤 ---
THRESHOLD = 0.4
filtered_annotations = []
count_before = len(raw_preds)

print(f"🔪 举起屠刀，开始清除低于 {THRESHOLD} 的垃圾框...")
for pred in raw_preds:
    score = pred.get('score', 0.0)
    # 只有置信度大于等于 0.4 的“黄金真理”才能活下来
    if score >= THRESHOLD:
        w, h = pred['bbox'][2], pred['bbox'][3]
        ann = {
            'id': len(filtered_annotations) + 1,
            'image_id': pred['image_id'],
            'category_id': pred['category_id'],
            'bbox': pred['bbox'],
            'area': w * h,
            'iscrowd': 0,
            'score': score
        }
        filtered_annotations.append(ann)

print("=========================================")
print(f"🛑 过滤前总框数: {count_before}")
print(f"✅ 过滤后保留数: {len(filtered_annotations)}")
print(f"🗑️ 成功砍掉垃圾: {count_before - len(filtered_annotations)} 个")
print("=========================================")

coco_dict['annotations'] = filtered_annotations
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(coco_dict, f)
print(f"💾 真正的纯净版高精伪标签已保存至: {out_path}")
