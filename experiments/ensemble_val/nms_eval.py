import json
from collections import defaultdict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ensemble_boxes import nms
from mmengine.config import Config
import os

print("🚀 正在自动解析验证集 Ground Truth 路径...")
cfg = Config.fromfile('external_modules/mmdetection/configs/grounding_dino/grounding_dino_swin-b_semi-sup_zerowaste.py')
gt_path = cfg.val_dataloader.dataset.ann_file
if hasattr(cfg, 'data_root') and not gt_path.startswith('/'):
    gt_path = os.path.join(cfg.data_root, gt_path)

coco_gt = COCO(gt_path)

print("📦 正在加载两份巅峰权重的预测结果...")
with open('experiments/ensemble_val/round1_preds.bbox.json', 'r') as f:
    preds1 = json.load(f)
with open('experiments/ensemble_val/round2_preds.bbox.json', 'r') as f:
    preds2 = json.load(f)

img_to_preds1 = defaultdict(list)
for p in preds1: img_to_preds1[p['image_id']].append(p)
img_to_preds2 = defaultdict(list)
for p in preds2: img_to_preds2[p['image_id']].append(p)

final_preds = []
print("🔥 正在执行 Ensemble NMS (不再搞平均主义啦，强者为尊！)...")

for img_id in coco_gt.imgs.keys():
    img_info = coco_gt.imgs[img_id]
    w, h = img_info['width'], img_info['height']
    
    boxes_list, scores_list, labels_list = [], [], []
    
    for preds in [img_to_preds1[img_id], img_to_preds2[img_id]]:
        boxes, scores, labels = [], [], []
        for p in preds:
            bx, by, bw, bh = p['bbox']
            x1, y1 = max(0.0, min(1.0, bx/w)), max(0.0, min(1.0, by/h))
            x2, y2 = max(0.0, min(1.0, (bx+bw)/w)), max(0.0, min(1.0, (by+bh)/h))
            if x2 <= x1 or y2 <= y1: continue
            boxes.append([x1, y1, x2, y2])
            scores.append(p['score'])
            labels.append(p['category_id'])
            
        boxes_list.append(boxes)
        scores_list.append(scores)
        labels_list.append(labels)
        
    if not boxes_list[0] and not boxes_list[1]: continue
        
    # 🌟 杀手锏：换成 NMS！保留各路专家的真实置信度
    boxes, scores, labels = nms(
        boxes_list, scores_list, labels_list, 
        weights=[1.0, 1.05], # 依然稍微偏袒一点点总分更高的 Round 2
        iou_thr=0.55
    )
    
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        bx, by = x1*w, y1*h
        bw, bh = (x2-x1)*w, (y2-y1)*h
        final_preds.append({
            'image_id': img_id,
            'category_id': int(label),
            'bbox': [bx, by, bw, bh],
            'score': float(score)
        })

out_json = 'experiments/ensemble_val/nms_preds.bbox.json'
with open(out_json, 'w') as f:
    json.dump(final_preds, f)
    
print(f"✅ NMS融合完毕，生成了 {len(final_preds)} 个预测框！")
print("📊 正在调用 COCO API 重新算分，准备见证真正的奇迹...")

coco_dt = coco_gt.loadRes(out_json)
coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
