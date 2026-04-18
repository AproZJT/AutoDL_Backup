import json
import os

gt_path = './data/zerowaste-f/train/labels.json'
pseudo_path = './data/zerowaste-f/teacher_pseudo_labels_high_prec.json'
clean_merged_path = './data/zerowaste-f/merged_train_labels_ultra_clean.json'

print("🔪 正在执行刮骨疗毒：切除金属伪标签 & 0.85 极限过滤...")

with open(gt_path, 'r', encoding='utf-8') as f:
    gt_data = json.load(f)
with open(pseudo_path, 'r', encoding='utf-8') as f:
    pseudo_data = json.load(f)

# 建立类别 ID 到名称的映射
cat_map = {cat['id']: cat['name'] for cat in pseudo_data.get('categories', [])}

valid_pseudo_anns = []
dropped_metal = 0
dropped_low_score = 0

for ann in pseudo_data['annotations']:
    cat_name = cat_map.get(ann['category_id'], '')
    
    # 1. 绝对切除：扔掉所有伪标签里的 metal
    if cat_name == 'metal':
        dropped_metal += 1
        continue
        
    # 2. 极限提纯：如果有 score 字段，低于 0.85 直接扔掉
    score = ann.get('score', 1.0)
    if score < 0.85:
        dropped_low_score += 1
        continue
        
    valid_pseudo_anns.append(ann)

# 3. 缝合到真值 (处理 ID 冲突)
max_img_id = max([img['id'] for img in gt_data['images']]) if gt_data['images'] else 0
max_ann_id = max([ann['id'] for ann in gt_data['annotations']]) if gt_data['annotations'] else 0

img_id_mapping = {}
for img in pseudo_data['images']:
    old_id = img['id']
    max_img_id += 1
    new_id = max_img_id
    img_id_mapping[old_id] = new_id
    new_img = img.copy()
    new_img['id'] = new_id
    gt_data['images'].append(new_img)

for ann in valid_pseudo_anns:
    max_ann_id += 1
    new_ann = ann.copy()
    new_ann['id'] = max_ann_id
    new_ann['image_id'] = img_id_mapping.get(ann['image_id'], ann['image_id'])
    # 移除 score 字段以符合 COCO 训练标准
    new_ann.pop('score', None)
    gt_data['annotations'].append(new_ann)

with open(clean_merged_path, 'w', encoding='utf-8') as f:
    json.dump(gt_data, f)

print(f"✅ 清洗完成！")
print(f"🗑️ 剔除有毒 Metal 伪标签: {dropped_metal} 个")
print(f"🗑️ 剔除低置信度伪标签: {dropped_low_score} 个")
print(f"✨ 最终保留的纯净伪标签加入训练集。账本已保存至: {clean_merged_path}")
