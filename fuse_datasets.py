import json
import os

gt_path = './data/zerowaste-f/train/labels.json'
pseudo_path = './data/zerowaste-f/teacher_pseudo_labels_high_prec.json'
merged_path = './data/zerowaste-f/merged_train_labels.json'

print("🔄 正在物理融合真值与伪标签...")

with open(gt_path, 'r', encoding='utf-8') as f:
    gt_data = json.load(f)
with open(pseudo_path, 'r', encoding='utf-8') as f:
    pseudo_data = json.load(f)

# 1. 寻找最大的 image_id 和 annotation_id 防止冲突
max_img_id = max([img['id'] for img in gt_data['images']]) if gt_data['images'] else 0
max_ann_id = max([ann['id'] for ann in gt_data['annotations']]) if gt_data['annotations'] else 0

# 2. 映射伪标签的 ID 并追加
img_id_mapping = {}
for img in pseudo_data['images']:
    old_id = img['id']
    max_img_id += 1
    new_id = max_img_id
    img_id_mapping[old_id] = new_id
    
    new_img = img.copy()
    new_img['id'] = new_id
    gt_data['images'].append(new_img)

for ann in pseudo_data['annotations']:
    max_ann_id += 1
    new_ann = ann.copy()
    new_ann['id'] = max_ann_id
    # 修正 image_id
    new_ann['image_id'] = img_id_mapping.get(ann['image_id'], ann['image_id'])
    gt_data['annotations'].append(new_ann)

with open(merged_path, 'w', encoding='utf-8') as f:
    json.dump(gt_data, f)

print(f"✅ 物理缝合完成！总图片数: {len(gt_data['images'])}, 总标注数: {len(gt_data['annotations'])}")
print(f"💾 缝合账本已保存至: {merged_path}")
