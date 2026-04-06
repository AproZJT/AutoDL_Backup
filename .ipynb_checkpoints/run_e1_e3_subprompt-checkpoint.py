import os
import json
import torch
import numpy as np
import warnings
from pathlib import Path
from tqdm import tqdm
from mmdet.apis import init_detector, inference_detector
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ensemble_boxes import weighted_boxes_fusion

warnings.filterwarnings("ignore")

# ========================================================
# 🎮 实验控制台
# USE_WBF = False -> 跑 E1 (子类 Prompt + 官方原生置信度)
# USE_WBF = True  -> 跑 E3 (子类 Prompt + WBF 融合)
# ========================================================
USE_WBF = False  

# ================= 1. 基础配置 =================
config_file = 'external_modules/mmdetection/configs/grounding_dino/grounding_dino_swin-b_inference_zerowaste_f.py'
checkpoint_file = 'weights/final_sota/best_swa_0.545.pth'
gt_path = 'data/zerowaste-f/test/labels.json' 
img_dir = 'data/zerowaste-f/test'

# 细粒度提示词 (9个子类)
text_prompt = "rigid plastic . hard plastic container . cardboard . corrugated carton . metal . metal can . aluminum foil . soft plastic . plastic bag ."

# 映射字典：将 0~8 映射回 COCO JSON 里真实的 category_id
MAIN_CLASSES = ['rigid_plastic', 'cardboard', 'metal', 'soft_plastic']

def main():
    print(f"\n[🚀 当前运行] 细粒度 Prompt验证 | WBF开启状态: {USE_WBF}")
    coco = COCO(gt_path)
    cats = coco.loadCats(coco.getCatIds())
    name2id = {c['name']: c['id'] for c in cats}
    
    # 建立 Grounding DINO 输出 index 到真实 category_id 的映射
    # 0,1->rigid; 2,3->cardboard; 4,5,6->metal; 7,8->soft
    SUB_TO_REAL_ID_MAP = {
        0: name2id['rigid_plastic'], 1: name2id['rigid_plastic'],
        2: name2id['cardboard'],     3: name2id['cardboard'],
        4: name2id['metal'],         5: name2id['metal'],         6: name2id['metal'],
        7: name2id['soft_plastic'],  8: name2id['soft_plastic']
    }

    print("\n[INFO] 正在加载 SOTA 模型...")
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    image_locator = {p.name: str(p) for p in Path("data").rglob("*") if p.is_file() and p.suffix.lower() in ['.png', '.jpg', '.jpeg']}
    
    results = []
    
    print("\n🔍 开始子类别推理...")
    for img_id in tqdm(coco.getImgIds()):
        img_info = coco.loadImgs(img_id)[0]
        W, H = img_info['width'], img_info['height']
        
        img_name = os.path.basename(img_info['file_name'])
        img_path = image_locator.get(img_name)
        if not img_path:
            guess1 = os.path.join(img_dir, img_name)
            img_path = guess1 if os.path.exists(guess1) else os.path.join(img_dir, 'data', img_name)
            if not os.path.exists(img_path): continue
            
        result_raw = inference_detector(model, img_path, text_prompt=text_prompt)
        pred = result_raw.pred_instances
        pred = pred[pred.scores > 0.01] 
        
        # 分组预测结果
        grouped_preds = {k: {'boxes':[], 'scores':[]} for k in name2id.values()}
        
        for i in range(len(pred)):
            sub_label = pred.labels[i].item()
            if sub_label not in SUB_TO_REAL_ID_MAP: continue
            
            real_cat_id = SUB_TO_REAL_ID_MAP[sub_label]
            bx1, by1, bx2, by2 = pred.bboxes[i].cpu().numpy()
            grouped_preds[real_cat_id]['boxes'].append([bx1, by1, bx2, by2])
            grouped_preds[real_cat_id]['scores'].append(pred.scores[i].item())

        for real_cat_id, data in grouped_preds.items():
            if not data['boxes']: continue
            
            boxes_np = np.array(data['boxes'], dtype=np.float32)
            scores_np = np.array(data['scores'], dtype=np.float32)
            
            if USE_WBF:
                boxes_norm = np.clip(boxes_np / np.array([W, H, W, H], dtype=np.float32), 0.0, 1.0)
                f_boxes, f_scores, _ = weighted_boxes_fusion(
                    [boxes_norm.tolist()], [scores_np.tolist()], [[real_cat_id] * len(scores_np)], 
                    weights=None, iou_thr=0.55, skip_box_thr=0.01
                )
                final_boxes = f_boxes * np.array([W, H, W, H], dtype=np.float32)
                final_scores = f_scores
            else:
                # 不使用 WBF 时，直接输出所有子类别的预测框
                final_boxes = boxes_np
                final_scores = scores_np
            
            for idx in range(len(final_boxes)):
                bx1, by1, bx2, by2 = final_boxes[idx]
                results.append({
                    "image_id": img_id,
                    "category_id": real_cat_id,
                    "bbox": [float(bx1), float(by1), float(bx2 - bx1), float(by2 - by1)],
                    "score": float(final_scores[idx])
                })

    out_json = f"ablation_E{'3' if USE_WBF else '1'}_results.json"
    with open(out_json, 'w') as f: json.dump(results, f)

    print(f"\n📊 正在生成最终成绩单 ({out_json})...")
    coco_dt = coco.loadRes(out_json)
    coco_eval = COCOeval(coco, coco_dt, 'bbox')
    coco_eval.params.maxDets = [100, 300, 1000] 
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

if __name__ == "__main__":
    main()