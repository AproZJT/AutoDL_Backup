import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# ======================================================
# Ensemble-Based Soft Pseudo-Labeling (Improved)
# ======================================================
# Key upgrades over the previous version:
# 1) Better IoU clustering: assign to BEST matching cluster, not first match.
# 2) Robust WBF score: weighted score + spread penalty + model-agreement bonus.
# 3) Per-model reliability weighting support.
# 4) Soft-NMS postprocess to suppress duplicates while keeping recall.
# 5) Size / aspect-ratio quality filters for noisy pseudo labels.
# ======================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Generate ensemble consensus pseudo annotations")
    parser.add_argument(
        "--input",
        type=str,
        default=str(Path("data") / "pseudo_labels" / "zerowaste-s_consolidated_pseudo_annotations.json"),
        help="Input consolidated COCO json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path("data") / "pseudo_labels" / "zerowaste-s_ensemble_consensus_pseudo_annotations.json"),
        help="Output ensemble COCO json",
    )

    # thresholds
    parser.add_argument("--tau", type=float, default=0.05, help="Initial score threshold")
    parser.add_argument("--tau_f", type=float, default=0.35, help="Final score threshold")
    parser.add_argument("--theta", type=float, default=0.65, help="IoU threshold for clustering")
    parser.add_argument("--min_votes", type=int, default=2, help="Minimum distinct model agreement")

    # consensus parameters
    parser.add_argument("--alpha", type=float, default=0.1, help="Spread decay factor")
    parser.add_argument("--beta", type=float, default=0.05, help="Model agreement bonus factor")
    parser.add_argument("--gamma", type=float, default=0.2, help="Cluster size bonus factor")

    # post filtering
    parser.add_argument("--min_area", type=float, default=16.0, help="Minimum bbox area to keep")
    parser.add_argument("--max_ar", type=float, default=12.0, help="Maximum aspect ratio allowed")

    # soft-nms
    parser.add_argument("--soft_nms_iou", type=float, default=0.55, help="Soft-NMS IoU threshold")
    parser.add_argument("--soft_nms_sigma", type=float, default=0.5, help="Soft-NMS sigma")
    parser.add_argument("--soft_nms_min_score", type=float, default=0.2, help="Soft-NMS minimum score")

    # Optional model reliability weights in JSON string format:
    # '{"model_0":1.0, "model_1":1.05, "model_2":0.95, "model_3":1.1}'
    parser.add_argument(
        "--model_weights_json",
        type=str,
        default="",
        help="Optional per-model reliability weights as JSON string",
    )

    return parser.parse_args()


def load_coco_json(json_path: str) -> Dict:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "annotations" not in data or "images" not in data or "categories" not in data:
        raise ValueError(f"Invalid COCO JSON format: Missing required keys in {json_path}")

    return data


def bbox_xywh_to_xyxy(box: List[float]) -> Tuple[float, float, float, float]:
    x, y, w, h = box
    return x, y, x + w, y + h


def bbox_area(box: List[float]) -> float:
    _, _, w, h = box
    if w <= 0 or h <= 0:
        return 0.0
    return float(w * h)


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    x1_1, y1_1, x2_1, y2_1 = bbox_xywh_to_xyxy(box1)
    x1_2, y1_2, x2_2, y2_2 = bbox_xywh_to_xyxy(box2)

    if x2_1 <= x1_1 or y2_1 <= y1_1 or x2_2 <= x1_2 or y2_2 <= y1_2:
        return 0.0

    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - inter_area

    return float(inter_area / union) if union > 0 else 0.0


def clip_bbox(bbox: List[float], img_width: float, img_height: float) -> List[float]:
    x, y, w, h = bbox
    x = max(0.0, min(x, img_width - 1))
    y = max(0.0, min(y, img_height - 1))
    w = max(0.0, min(w, img_width - x))
    h = max(0.0, min(h, img_height - y))
    return [x, y, w, h]


def is_reasonable_box(bbox: List[float], min_area: float, max_ar: float) -> bool:
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return False
    area = w * h
    if area < min_area:
        return False
    ar = max(w / h, h / w)
    if ar > max_ar:
        return False
    if x < 0 or y < 0:
        return False
    return True


def organize_predictions(annotations: List[Dict], tau: float) -> Dict[int, Dict[int, List[Dict]]]:
    preds = defaultdict(lambda: defaultdict(list))
    for ann in annotations:
        if "image_id" not in ann or "category_id" not in ann or "bbox" not in ann:
            continue

        score = float(ann.get("score", 0.0))
        if score < tau:
            continue

        ann["score"] = score
        preds[int(ann["image_id"])][int(ann["category_id"])].append(ann)

    return preds


def compute_cluster_representative(cluster: List[Dict]) -> List[float]:
    # use score-weighted average as representative; more stable than first/highest only
    boxes = np.array([p["bbox"] for p in cluster], dtype=np.float32)
    scores = np.array([max(1e-6, p["score"]) for p in cluster], dtype=np.float32)
    weights = scores / scores.sum()
    rep = np.dot(weights, boxes)
    return rep.tolist()


def build_iou_clusters(predictions: List[Dict], iou_threshold: float) -> List[List[Dict]]:
    # sort by score descending for deterministic behavior
    sorted_preds = sorted(predictions, key=lambda p: p["score"], reverse=True)
    clusters: List[List[Dict]] = []

    for pred in sorted_preds:
        best_idx = -1
        best_iou = 0.0

        for ci, cluster in enumerate(clusters):
            rep_bbox = compute_cluster_representative(cluster)
            iou = calculate_iou(pred["bbox"], rep_bbox)
            if iou > best_iou:
                best_iou = iou
                best_idx = ci

        if best_idx >= 0 and best_iou >= iou_threshold:
            clusters[best_idx].append(pred)
        else:
            clusters.append([pred])

    return clusters


def weighted_box_fusion(cluster: List[Dict], model_weights: Dict[str, float]) -> Tuple[List[float], float]:
    boxes = np.array([pred["bbox"] for pred in cluster], dtype=np.float32)
    scores = np.array([pred["score"] for pred in cluster], dtype=np.float32)

    effective_scores = []
    for pred, s in zip(cluster, scores):
        mw = model_weights.get(pred.get("model_id", ""), 1.0)
        effective_scores.append(max(1e-6, float(s * mw)))

    effective_scores = np.array(effective_scores, dtype=np.float32)
    weights = effective_scores / effective_scores.sum()

    fused_box = np.dot(weights, boxes).tolist()

    # robust base score: emphasize strong detections, but preserve consensus signal
    top = float(np.max(effective_scores))
    mean = float(np.mean(effective_scores))
    p75 = float(np.percentile(effective_scores, 75))
    base_score = 0.5 * top + 0.3 * p75 + 0.2 * mean

    return fused_box, base_score


def consensus_factor(
    cluster: List[Dict],
    fused_box: List[float],
    alpha: float,
    beta: float,
    gamma: float,
) -> float:
    ious = [calculate_iou(pred["bbox"], fused_box) for pred in cluster]
    avg_iou = float(np.mean(ious)) if ious else 0.0

    # spread penalty (lower avg_iou => larger penalty)
    spread = 1.0 - avg_iou
    agreement = float(np.exp(-alpha * spread))

    # model agreement bonus
    model_ids = {pred.get("model_id", "") for pred in cluster if pred.get("model_id", "")}
    num_models = len(model_ids)
    model_bonus = min(1.20, 1.0 + beta * max(0, num_models - 1))

    # cluster-size bonus (mild)
    cluster_bonus = min(1.15, 1.0 + gamma * max(0, len(cluster) - 2) / max(1, len(cluster)))

    return agreement * model_bonus * cluster_bonus


def fuse_cluster(
    cluster: List[Dict],
    alpha: float,
    beta: float,
    gamma: float,
    model_weights: Dict[str, float],
) -> Dict:
    fused_box, base_score = weighted_box_fusion(cluster, model_weights)
    c_factor = consensus_factor(cluster, fused_box, alpha, beta, gamma)
    final_score = base_score * c_factor

    model_ids = sorted(list({pred.get("model_id", "") for pred in cluster if pred.get("model_id", "")}))
    rep = max(cluster, key=lambda p: p["score"])

    return {
        "image_id": int(rep["image_id"]),
        "category_id": int(rep["category_id"]),
        "bbox": fused_box,
        "score": float(final_score),
        "model_ids": model_ids,
        "num_models": len(model_ids),
    }


def soft_nms_for_class(
    preds: List[Dict],
    iou_thr: float,
    sigma: float,
    min_score: float,
) -> List[Dict]:
    # Gaussian Soft-NMS
    candidates = [p.copy() for p in preds]
    kept = []

    while candidates:
        candidates.sort(key=lambda x: x["score"], reverse=True)
        best = candidates.pop(0)
        kept.append(best)

        survivors = []
        for p in candidates:
            iou = calculate_iou(best["bbox"], p["bbox"])
            if iou > iou_thr:
                p["score"] = float(p["score"] * np.exp(-(iou * iou) / max(1e-6, sigma)))
            if p["score"] >= min_score:
                survivors.append(p)
        candidates = survivors

    return kept


def main():
    args = parse_args()

    model_weights = {}
    if args.model_weights_json.strip():
        model_weights = json.loads(args.model_weights_json)

    coco_data = load_coco_json(args.input)
    annotations = coco_data["annotations"]
    images = coco_data["images"]
    categories = coco_data["categories"]

    predictions_by_image = organize_predictions(annotations, tau=args.tau)
    print(f"Initial filtered predictions (tau={args.tau}): {sum(len(v2) for v1 in predictions_by_image.values() for v2 in v1.values())}")
    print(f"Processed images: {len(predictions_by_image)}")

    # Build image dimension lookup with fallback defaults
    image_dimensions = {
        int(img["id"]): (float(img.get("width", 1920)), float(img.get("height", 1080))) for img in images
    }

    total_images = len(predictions_by_image)
    processed_images_counter = 0
    progress_step = 500
    progress_start_time = time.time()

    fused_all = []
    for img_id, class_dict in predictions_by_image.items():
        processed_images_counter += 1
        if processed_images_counter % progress_step == 0 or processed_images_counter == total_images:
            elapsed = time.time() - progress_start_time
            pct = (processed_images_counter / max(1, total_images)) * 100.0
            print(
                f"[IMG_PROGRESS] processed={processed_images_counter}/{total_images} ({pct:.1f}%) | elapsed={elapsed:.1f}s",
                flush=True,
            )
        for class_id, preds in class_dict.items():
            clusters = build_iou_clusters(preds, iou_threshold=args.theta)

            for cluster in clusters:
                distinct_models = {p.get("model_id", "") for p in cluster if p.get("model_id", "")}
                if len(distinct_models) < args.min_votes:
                    continue

                fused = fuse_cluster(
                    cluster=cluster,
                    alpha=args.alpha,
                    beta=args.beta,
                    gamma=args.gamma,
                    model_weights=model_weights,
                )

                w, h = image_dimensions.get(img_id, (1920.0, 1080.0))
                fused["bbox"] = clip_bbox(fused["bbox"], w, h)

                if not is_reasonable_box(fused["bbox"], args.min_area, args.max_ar):
                    continue

                fused_all.append(fused)

    # Soft-NMS per (image_id, category_id)
    grouped = defaultdict(list)
    for p in fused_all:
        grouped[(p["image_id"], p["category_id"])].append(p)

    post_nms = []
    for key, preds in grouped.items():
        kept = soft_nms_for_class(
            preds,
            iou_thr=args.soft_nms_iou,
            sigma=args.soft_nms_sigma,
            min_score=args.soft_nms_min_score,
        )
        post_nms.extend(kept)

    # final threshold
    final_predictions = [p for p in post_nms if p["score"] >= args.tau_f]

    # assign annotation IDs + area + iscrowd for cleaner COCO compatibility
    for idx, pred in enumerate(final_predictions, start=1):
        pred["id"] = idx
        pred["area"] = bbox_area(pred["bbox"])
        pred["iscrowd"] = 0

    final_coco = {
        "images": images,
        "categories": categories,
        "annotations": final_predictions,
    }

    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_coco, f, indent=2)

    print(f"Saved ensemble pseudo-labels: {output_file}")
    print(f"Final annotations count: {len(final_predictions)}")


if __name__ == "__main__":
    main()
