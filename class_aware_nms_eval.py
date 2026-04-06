import os
import json
import torch
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

from tqdm import tqdm
from torchvision.ops import nms
from mmdet.apis import init_detector, inference_detector
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ================= 1. 核心配置 =================
config_file = 'external_modules/mmdetection/configs/grounding_dino/grounding_dino_swin-b_inference_zerowaste_f.py'
checkpoint_file = 'weights/final_sota/best_swa_0.545.pth'

# 真实的测试集 JSON
gt_path_config = 'data/zerowaste-f/test/labels.json' 
img_dir = 'data/zerowaste-f/test'

# 严格的类别名称声明 (必须与 COCO JSON 里的 name 保持一致)
class_names = ['rigid_plastic', 'cardboard', 'metal', 'soft_plastic']
text_prompt = "rigid plastic . cardboard . metal . soft plastic ."

# ================= 2. 物理属性定制 IoU 阈值 =================
CLASS_IOU_THRESH = {
    0: 0.45,  # rigid_plastic: 低阈值强力去重
    1: 0.80,  # cardboard: 高阈值保护堆叠纸板
    2: 0.55,  # metal: 常规阈值
    3: 0.65   # soft_plastic: 允许轻度挤压堆叠
}

def main():
    print(f"\n[INFO] 🎯 锁定真实测试集 GT: {gt_path_config}")
    
    # 1. 加载 COCO 标注
    coco = COCO(gt_path_config)
    
    # 2. 【核心修复】精准字符串对齐，拒绝“答题卡涂串”！
    cats = coco.loadCats(coco.getCatIds())
    name2id = {c['name']: c['id'] for c in cats}
    try:
        id2cat = {i: name2id[name] for i, name in enumerate(class_names)}
        print(f"[INFO] 类别对齐成功: {id2cat}")
    except KeyError as e:
        print(f"[ERROR] JSON里的类别名与代码对不上！JSON包含: {name2id.keys()}")
        raise e

    print("\n🚀 正在加载 SOTA 融合模型 (0.545)...")
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    img_ids = coco.getImgIds()
    results = []

    # 3. 图像全盘雷达索引
    print("\n[INFO] 正在建立图像文件雷达索引...")
    image_locator = {}
    for p in Path("data").rglob("*"):
        if p.is_file() and p.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            image_locator[p.name] = str(p)

    # 4. 推理与物理后处理
    print("\n🔍 开始类感知 NMS 推理...")
    for img_id in tqdm(img_ids):
        img_info = coco.loadImgs(img_id)[0]
        img_name = os.path.basename(img_info['file_name'])
        
        img_path = image_locator.get(img_name)
        if not img_path:
            guess1 = os.path.join(img_dir, img_name)
            guess2 = os.path.join(img_dir, 'data', img_name)
            img_path = guess1 if os.path.exists(guess1) else guess2
            if not os.path.exists(img_path):
                continue
        
        # 获取原始推理结果
        result_raw = inference_detector(model, img_path, text_prompt=text_prompt)
        pred = result_raw.pred_instances
        
        # 【对齐官方评估】过滤极低置信度并截取 Top 300
        pred = pred[pred.scores > 0.01]
        if len(pred) > 300:
            top_k_idx = torch.topk(pred.scores, 300)[1]
            pred = pred[top_k_idx]
        
        # 按类别进行切割与 NMS
        for cls_idx in range(4):
            mask = pred.labels == cls_idx
            cls_boxes = pred.bboxes[mask]
            cls_scores = pred.scores[mask]
            
            if len(cls_boxes) == 0:
                continue
                
            # 魔法生效点：带入物理属性阈值
            keep_indices = nms(cls_boxes, cls_scores, iou_threshold=CLASS_IOU_THRESH[cls_idx])
            
            final_boxes = cls_boxes[keep_indices].cpu().numpy()
            final_scores = cls_scores[keep_indices].cpu().numpy()
            
            for i in range(len(final_boxes)):
                bx1, by1, bx2, by2 = final_boxes[i]
                results.append({
                    "image_id": img_id,
                    "category_id": id2cat[cls_idx], # 精准映射！
                    "bbox": [float(bx1), float(by1), float(bx2 - bx1), float(by2 - by1)],
                    "score": float(final_scores[i])
                })

    # 5. COCO 官方评估
    out_json = "class_aware_nms_results.json"
    with open(out_json, 'w') as f:
        json.dump(results, f)

    print("\n📊 正在生成最终成绩单...")
    coco_dt = coco.loadRes(out_json)
    coco_eval = COCOeval(coco, coco_dt, 'bbox')
    
    # 完美对齐官方输出格式 (100, 300, 1000)
    coco_eval.params.maxDets = [100, 300, 1000] 
    
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

if __name__ == "__main__":
    main()