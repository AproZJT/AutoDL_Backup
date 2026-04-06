import os
import json
import warnings
from typing import List, Tuple

warnings.filterwarnings("ignore")

from tqdm import tqdm
from mmdet.apis import init_detector, inference_detector
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ensemble_boxes import weighted_boxes_fusion

# ================= 配置路径 =================
config_file = 'external_modules/mmdetection/configs/grounding_dino/grounding_dino_swin-b_inference_zerowaste_f.py'
checkpoint_file = 'weights/gdino-swin-b/zerowaste_semi-sup_best_coco_bbox_mAP.pth'
gt_path = 'data/zerowaste-f/test/labels.json'
img_dir = 'data/zerowaste-f/test/data'

# ================= 基础类别顺序（必须与模型输出 label 对齐） =================
base_classes: Tuple[str, ...] = (
    'rigid_plastic',
    'cardboard',
    'metal',
    'soft_plastic',
)

# ================= 弹药库：5组 Prompt（语义融合版） =================
prompt_sets: List[Tuple[str, str, str, str]] = [
    # baseline：保留模型“肌肉记忆”
    ('rigid plastic', 'cardboard', 'metal', 'soft plastic'),

    # Base + 扩写：在不丢失基础语义的情况下补充描述
    ('rigid plastic, hard plastic bottle', 'cardboard, corrugated cardboard box', 'metal, aluminum can', 'soft plastic, plastic bag'),
    ('rigid plastic, rigid plastic container', 'cardboard, flattened cardboard', 'metal, crushed metal can', 'soft plastic, clear plastic film'),
    ('rigid plastic, clear plastic bottle', 'cardboard, brown cardboard piece', 'metal, metal tin container', 'soft plastic, crumpled plastic wrapper'),
    ('rigid plastic, hard plastic packaging', 'cardboard, paperboard packaging', 'metal, scrap metal piece', 'soft plastic, soft plastic packaging'),
]

# 给第一组 baseline 更高权重
weights = [1.2, 1.0, 1.0, 1.0, 1.0]


def check_paths():
    for p in [config_file, checkpoint_file, gt_path, img_dir]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Path not found: {p}")


def clip01(v: float) -> float:
    return max(0.0, min(1.0, v))


def to_norm_xyxy(box, w: float, h: float):
    x1, y1, x2, y2 = box
    nx1 = clip01(float(x1) / max(1.0, w))
    ny1 = clip01(float(y1) / max(1.0, h))
    nx2 = clip01(float(x2) / max(1.0, w))
    ny2 = clip01(float(y2) / max(1.0, h))
    if nx2 <= nx1 or ny2 <= ny1:
        return None
    return [nx1, ny1, nx2, ny2]


def main():
    check_paths()

    print("1. 正在加载 GroundingDINO Swin-B 模型 (请稍候)...")
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    print("2. 正在加载真实标签数据...")
    coco = COCO(gt_path)
    img_ids = coco.getImgIds()

    # 建立 name -> category_id 映射，避免 cat_ids 顺序不一致
    categories = coco.loadCats(coco.getCatIds())
    name_to_catid = {c['name']: c['id'] for c in categories}
    class_to_catid = {name: name_to_catid[name] for name in base_classes if name in name_to_catid}

    if len(class_to_catid) != len(base_classes):
        raise ValueError(
            f"COCO categories mismatch. expected={base_classes}, got={[c['name'] for c in categories]}"
        )

    all_fused_results = []

    print(f"3. 开始多路 Prompt 推理与 WBF 融合 (共 {len(img_ids)} 张图片)...")
    for img_id in tqdm(img_ids):
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, img_info['file_name'])

        if not os.path.exists(img_path):
            continue

        w, h = float(img_info['width']), float(img_info['height'])

        boxes_list, scores_list, labels_list = [], [], []

        # 同一张图，使用多组 prompt
        for prompts in prompt_sets:
            # GroundingDINO 需要通过 text_prompt 注入文本，而不是只改 dataset_meta
            text_prompt = ' . '.join(prompts)

            result = inference_detector(
                model,
                img_path,
                text_prompt=text_prompt,
                custom_entities=True,
            )
            pred = result.pred_instances
            # 收紧阈值，避免多 prompt 融合时低质框噪声堆积
            pred = pred[pred.scores > 0.20]

            boxes = pred.bboxes.cpu().numpy()
            scores = pred.scores.cpu().numpy()
            labels = pred.labels.cpu().numpy()

            norm_boxes, norm_scores, norm_labels = [], [], []
            for box, score, label in zip(boxes, scores, labels):
                nb = to_norm_xyxy(box, w, h)
                if nb is None:
                    continue
                norm_boxes.append(nb)
                norm_scores.append(float(score))
                norm_labels.append(int(label))

            boxes_list.append(norm_boxes)
            scores_list.append(norm_scores)
            labels_list.append(norm_labels)

        # 所有 prompt 都无结果则跳过
        if not any(len(b) > 0 for b in boxes_list):
            continue

        # WBF 融合
        f_boxes, f_scores, f_labels = weighted_boxes_fusion(
            boxes_list,
            scores_list,
            labels_list,
            weights=weights,
            iou_thr=0.55,
            skip_box_thr=0.01,
        )

        # 组装 COCO 检测结果
        for box, score, label in zip(f_boxes, f_scores, f_labels):
            x1, y1, x2, y2 = box
            abs_x, abs_y = x1 * w, y1 * h
            abs_w, abs_h = (x2 - x1) * w, (y2 - y1) * h

            label_idx = int(label)
            class_name = base_classes[label_idx]
            category_id = class_to_catid[class_name]

            all_fused_results.append(
                {
                    "image_id": int(img_id),
                    "category_id": int(category_id),
                    "bbox": [float(abs_x), float(abs_y), float(abs_w), float(abs_h)],
                    "score": float(score),
                }
            )

    out_json = "ensemble_fused_results.json"
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(all_fused_results, f)

    print("\n" + "=" * 50)
    print("4. 正在生成最终成绩单...")
    print("=" * 50)

    coco_dt = coco.loadRes(out_json)
    coco_eval = COCOeval(coco, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == "__main__":
    main()
