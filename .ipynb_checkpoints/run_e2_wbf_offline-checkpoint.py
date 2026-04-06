import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ensemble_boxes import weighted_boxes_fusion
from collections import defaultdict

# ================= 1. 核心路径配置 =================
gt_path = 'data/zerowaste-f/test/labels.json'
pred_path = 'sota_results.bbox.json'  # 你原生 0.545 的预测结果

print(f"\n[INFO] 🎯 加载真实标签以获取图像宽高: {gt_path}")
coco = COCO(gt_path)
img_info_dict = {img['id']: img for img in coco.dataset['images']}

def main():
    print(f"[INFO] 🚀 加载 0.545 满血预测数据: {pred_path}")
    with open(pred_path, 'r') as f:
        preds = json.load(f)

    # 按 image_id 和 category_id 分组
    grouped = defaultdict(lambda: defaultdict(list))
    for p in preds:
        grouped[p['image_id']][p['category_id']].append(p)

    final_preds = []
    print("\n🔍 正在进行纯离线 WBF 加权融合...")
    
    for img_id, cat_dict in grouped.items():
        W = img_info_dict[img_id]['width']
        H = img_info_dict[img_id]['height']
        
        for cat_id, boxes_list in cat_dict.items():
            if not boxes_list: continue

            boxes, scores = [], []
            for b in boxes_list:
                x, y, w, h = b['bbox']
                boxes.append([x, y, x + w, y + h])
                scores.append(b['score'])

            boxes_np = np.array(boxes, dtype=np.float32)
            scores_np = np.array(scores, dtype=np.float32)

            # 🛡️ 核心防坑：WBF 强制要求坐标归一化到 [0, 1]
            boxes_norm = boxes_np / np.array([W, H, W, H], dtype=np.float32)
            boxes_norm = np.clip(boxes_norm, 0.0, 1.0)

            boxes_list_wbf = [boxes_norm.tolist()]
            scores_list_wbf = [scores_np.tolist()]
            labels_list_wbf = [[cat_id] * len(scores_np)]

            # 执行 WBF (同模型内部融合，iou_thr 设置为 0.55 较稳)
            f_boxes, f_scores, f_labels = weighted_boxes_fusion(
                boxes_list_wbf, scores_list_wbf, labels_list_wbf, 
                weights=None, iou_thr=0.55, skip_box_thr=0.01
            )

            # 还原坐标并转换为 COCO [x, y, w, h] 格式
            final_boxes = f_boxes * np.array([W, H, W, H], dtype=np.float32)
            for idx in range(len(final_boxes)):
                bx1, by1, bx2, by2 = final_boxes[idx]
                final_preds.append({
                    "image_id": img_id,
                    "category_id": cat_id,
                    "bbox": [float(bx1), float(by1), float(bx2 - bx1), float(by2 - by1)],
                    "score": float(f_scores[idx])
                })

    out_json = "ablation_E2_wbf_results.json"
    with open(out_json, 'w') as f:
        json.dump(final_preds, f)

    print("\n📊 生成 E2 (WBF Only) 成绩单...")
    coco_dt = coco.loadRes(out_json)
    coco_eval = COCOeval(coco, coco_dt, 'bbox')
    coco_eval.params.maxDets = [100, 300, 1000] 
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

if __name__ == "__main__":
    main()