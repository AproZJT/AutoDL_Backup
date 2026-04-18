import json

shell_path = 'data/zerowaste-f/unlabeled/labels.json'
list_path = 'data/zerowaste-f/final_pseudo_labels.json'
out_path = 'data/zerowaste-f/final_pseudo_coco.json'

print("⏳ 正在读取数据...")
with open(shell_path, 'r') as f:
    coco_dict = json.load(f)

with open(list_path, 'r') as f:
    preds_list = json.load(f)

print(f"📦 发现 {len(preds_list)} 个伪标签预测框。正在拼装标准 COCO 格式...")

annotations = []
for idx, pred in enumerate(preds_list):
    # 提取 bbox (COCO 格式标准为 [x, y, width, height])
    bbox = pred["bbox"]
    area = bbox[2] * bbox[3] # 宽 * 高
    
    # 组装标准 COCO annotation 字段
    ann = {
        "id": idx + 1,
        "image_id": pred["image_id"],
        "category_id": pred["category_id"],
        "bbox": bbox,
        "area": area,
        "iscrowd": 0,
        "score": pred.get("score", 1.0) # 保留 Teacher 的置信度
    }
    annotations.append(ann)

# 写入大字典
coco_dict["annotations"] = annotations

print(f"💾 正在保存至 {out_path} ... (文件较大，请稍候)")
with open(out_path, 'w') as f:
    json.dump(coco_dict, f)

print(f"✅ 转换完成！最终的 COCO 训练格式伪标签已生成: {out_path}")
