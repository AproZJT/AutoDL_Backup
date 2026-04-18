import json

gt_path = './data/zerowaste-f/unlabeled/labels.json'
pseudo_path = './data/zerowaste-f/teacher_pseudo_labels.json'

print("🔄 正在读取数据，准备进行缝合...")
with open(gt_path, 'r', encoding='utf-8') as f:
    coco_dict = json.load(f)

with open(pseudo_path, 'r', encoding='utf-8') as f:
    pseudo_list = json.load(f)

print(f"📦 缝合中... 共提取到 {len(pseudo_list)} 个伪标签框。")

# 将 list 格式的预测框转换为标准 annotation 格式
annotations = []
for i, pred in enumerate(pseudo_list):
    # bbox 是 [x, y, w, h]
    w, h = pred['bbox'][2], pred['bbox'][3]
    ann = {
        'id': i + 1,
        'image_id': pred['image_id'],
        'category_id': pred['category_id'],
        'bbox': pred['bbox'],
        'area': w * h,          # COCO 格式强依赖 area 进行大小目标划分
        'iscrowd': 0,           # 默认非人群遮挡
        'score': pred.get('score', 1.0) # 保留置信度
    }
    annotations.append(ann)

# 替换原版的 annotations
coco_dict['annotations'] = annotations

# 覆写保存
with open(pseudo_path, 'w', encoding='utf-8') as f:
    json.dump(coco_dict, f)

print("✅ 缝合完成！伪标签已成功伪装成完整的 COCO 训练集格式！")
