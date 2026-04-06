import json
import os
import warnings
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")

from tqdm import tqdm
from mmdet.apis import init_detector, inference_detector
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ensemble_boxes import weighted_boxes_fusion

# ================== 路径配置（按你当前项目结构） ==================
config_file = "external_modules/mmdetection/configs/grounding_dino/grounding_dino_swin-b_inference_zerowaste_f.py"
checkpoint_file = "experiments/swin-b_HeadOnly_Finetune_FINAL/best_coco_bbox_mAP_epoch_2.pth"
gt_path = "data/zerowaste-f/test/labels.json"
img_dir = "data/zerowaste-f/test/data"
output_json = "patch_ensemble_fused_results.json"

# ================== 参数 ==================
SCORE_THRESH = 0.15
patch_size = 960
patch_stride = 640
WBF_IOU_THR = 0.55
WBF_SKIP_BOX_THR = 0.05

# 模型类别顺序（必须与训练时一致）
base_classes: Tuple[str, ...] = (
    "rigid_plastic",
    "cardboard",
    "metal",
    "soft_plastic",
)


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


def xyxy_to_xywh(x1: float, y1: float, x2: float, y2: float):
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def generate_patches(width: int, height: int, size: int, stride: int):
    patches = []
    ys = list(range(0, max(1, height - size + 1), stride))
    xs = list(range(0, max(1, width - size + 1), stride))

    if not ys or ys[-1] != max(0, height - size):
        ys.append(max(0, height - size))
    if not xs or xs[-1] != max(0, width - size):
        xs.append(max(0, width - size))

    for y in ys:
        for x in xs:
            x2 = min(width, x + size)
            y2 = min(height, y + size)
            patches.append((x, y, x2, y2))
    return patches


def extract_preds(result, score_thr: float):
    pred = result.pred_instances
    pred = pred[pred.scores > score_thr]
    boxes = pred.bboxes.cpu().numpy()
    scores = pred.scores.cpu().numpy()
    labels = pred.labels.cpu().numpy()
    return boxes, scores, labels


def main():
    check_paths()

    print("1) 加载模型...")
    model = init_detector(config_file, checkpoint_file, device="cuda:0")

    print("2) 加载 COCO 标注...")
    coco = COCO(gt_path)
    img_ids = coco.getImgIds()

    # name -> category_id 映射（避免 cat_ids 顺序错位）
    cats = coco.loadCats(coco.getCatIds())
    name_to_id: Dict[str, int] = {c["name"]: c["id"] for c in cats}
    missing = [n for n in base_classes if n not in name_to_id]
    if missing:
        raise ValueError(f"Missing categories in annotation: {missing}")
    label_to_catid = [name_to_id[n] for n in base_classes]

    all_fused = []

    print(f"3) 开始全局+切片推理并融合，共 {len(img_ids)} 张图...")
    for img_id in tqdm(img_ids):
        info = coco.loadImgs(img_id)[0]
        file_name = info["file_name"]
        w, h = int(info["width"]), int(info["height"])
        img_path = os.path.join(img_dir, file_name)

        if not os.path.exists(img_path):
            continue

        # GroundingDINO 需要 text_prompt
        text_prompt = " . ".join(base_classes)

        # 切片推理依赖 cv2，同时为了兼容推理管道，这里统一使用 ndarray 输入
        try:
            import cv2
        except Exception:
            raise ImportError("cv2 is required for patch inference. Please install opencv-python.")

        img = cv2.imread(img_path)
        if img is None:
            continue

        # 全图预测（使用 ndarray，避免路径/管道类型不一致）
        global_res = inference_detector(
            model,
            img,
            text_prompt=text_prompt,
            custom_entities=True,
        )
        g_boxes, g_scores, g_labels = extract_preds(global_res, SCORE_THRESH)

        boxes_list: List[List[List[float]]] = []
        scores_list: List[List[float]] = []
        labels_list: List[List[int]] = []

        # 全图加入 WBF 输入
        g_norm_boxes, g_norm_scores, g_norm_labels = [], [], []
        for box, score, label in zip(g_boxes, g_scores, g_labels):
            nb = to_norm_xyxy(box, w, h)
            if nb is None:
                continue
            g_norm_boxes.append(nb)
            g_norm_scores.append(float(score))
            g_norm_labels.append(int(label))

        boxes_list.append(g_norm_boxes)
        scores_list.append(g_norm_scores)
        labels_list.append(g_norm_labels)

        # 切片预测

        patches = generate_patches(w, h, patch_size, patch_stride)

        for (x1, y1, x2, y2) in patches:
            patch = img[y1:y2, x1:x2]
            if patch.size == 0:
                continue

            patch_res = inference_detector(
                model,
                patch,
                text_prompt=text_prompt,
                custom_entities=True,
            )
            p_boxes, p_scores, p_labels = extract_preds(patch_res, SCORE_THRESH)

            norm_boxes_p, norm_scores_p, norm_labels_p = [], [], []
            for box, score, label in zip(p_boxes, p_scores, p_labels):
                px1, py1, px2, py2 = box
                abs_x1, abs_y1 = px1 + x1, py1 + y1
                abs_x2, abs_y2 = px2 + x1, py2 + y1

                nb = to_norm_xyxy([abs_x1, abs_y1, abs_x2, abs_y2], w, h)
                if nb is None:
                    continue
                norm_boxes_p.append(nb)
                norm_scores_p.append(float(score))
                norm_labels_p.append(int(label))

            boxes_list.append(norm_boxes_p)
            scores_list.append(norm_scores_p)
            labels_list.append(norm_labels_p)

        if not any(len(b) > 0 for b in boxes_list):
            continue

        f_boxes, f_scores, f_labels = weighted_boxes_fusion(
            boxes_list,
            scores_list,
            labels_list,
            weights=[1.0] * len(boxes_list),
            iou_thr=WBF_IOU_THR,
            skip_box_thr=WBF_SKIP_BOX_THR,
        )

        for i in range(len(f_boxes)):
            x1n, y1n, x2n, y2n = f_boxes[i]
            ax1, ay1 = x1n * w, y1n * h
            ax2, ay2 = x2n * w, y2n * h
            bbox = xyxy_to_xywh(ax1, ay1, ax2, ay2)

            lbl = int(f_labels[i])
            if lbl < 0 or lbl >= len(label_to_catid):
                continue

            all_fused.append(
                {
                    "image_id": int(img_id),
                    "category_id": int(label_to_catid[lbl]),
                    "bbox": [float(v) for v in bbox],
                    "score": float(f_scores[i]),
                }
            )

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_fused, f)

    print("\n4) COCOEval 评估...")
    coco_dt = coco.loadRes(output_json)
    coco_eval = COCOeval(coco, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    print(f"\nDone. Saved fused results to: {output_json}")


if __name__ == "__main__":
    main()
