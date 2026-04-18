import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def iou_xywh(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    ua = aw * ah + bw * bh - inter
    return inter / max(ua, 1e-12)


def nms_xywh(records: List[dict], iou_thr: float) -> List[dict]:
    if not records:
        return []

    boxes = np.array([r['bbox'] for r in records], dtype=np.float32)
    scores = np.array([r['score'] for r in records], dtype=np.float32)
    order = scores.argsort()[::-1]

    keep_idx = []
    while order.size > 0:
        i = order[0]
        keep_idx.append(i)

        suppressed = [0]
        for j in range(1, order.size):
            if iou_xywh(boxes[i], boxes[order[j]]) > iou_thr:
                suppressed.append(j)
        order = np.delete(order, suppressed)

    return [records[i] for i in keep_idx]


def build_maps(coco_gt: COCO) -> Tuple[Dict[str, int], Dict[int, str]]:
    cats = coco_gt.loadCats(coco_gt.getCatIds())
    name2id = {c['name']: int(c['id']) for c in cats}
    id2name = {int(c['id']): c['name'] for c in cats}
    return name2id, id2name


def bucket_predictions(raw_preds: List[dict]) -> Dict[Tuple[int, int], List[dict]]:
    bucket = {}
    for p in raw_preds:
        img_id = int(p['image_id'])
        cat_id = int(p['category_id'])
        p['score'] = float(p['score'])
        key = (img_id, cat_id)
        bucket.setdefault(key, []).append(p)
    return bucket


def filter_predictions(
    pred_bucket: Dict[Tuple[int, int], List[dict]],
    per_class_thr: Dict[int, float],
    per_class_nms_iou: Dict[int, float],
) -> List[dict]:
    filtered = []
    for (_, cat_id), preds in pred_bucket.items():
        thr = per_class_thr.get(cat_id, 0.0)
        iou_thr = per_class_nms_iou.get(cat_id, 1.0)

        kept = [p for p in preds if p['score'] >= thr]
        if iou_thr < 0.999:
            kept = nms_xywh(kept, iou_thr)

        filtered.extend(kept)
    return filtered


def evaluate(coco_gt: COCO, preds: List[dict], max_dets: List[int]) -> None:
    coco_dt = coco_gt.loadRes(preds)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.params.maxDets = max_dets
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


def parse_args():
    parser = argparse.ArgumentParser(description='Apply best postprocess params and evaluate/export')
    parser.add_argument('--gt_json', type=str, required=True)
    parser.add_argument('--pred_json', type=str, required=True)
    parser.add_argument('--out_json', type=str, required=True)

    parser.add_argument('--default_thr', type=float, default=0.0)
    parser.add_argument('--rigid_thr', type=float, default=0.0)
    parser.add_argument('--cardboard_thr', type=float, default=0.0)
    parser.add_argument('--metal_thr', type=float, default=0.0)
    parser.add_argument('--soft_thr', type=float, default=0.0)

    parser.add_argument('--default_nms_iou', type=float, default=1.0)
    parser.add_argument('--metal_nms_iou', type=float, default=0.7)

    parser.add_argument('--max_dets', type=str, default='100,300,1000')
    parser.add_argument('--skip_eval', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()

    gt_path = Path(args.gt_json)
    pred_path = Path(args.pred_json)
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    coco_gt = COCO(str(gt_path))
    name2id, _ = build_maps(coco_gt)

    with open(pred_path, 'r', encoding='utf-8') as f:
        raw_preds = json.load(f)

    pred_bucket = bucket_predictions(raw_preds)

    per_class_thr = {cid: args.default_thr for cid in coco_gt.getCatIds()}
    if 'rigid_plastic' in name2id:
        per_class_thr[name2id['rigid_plastic']] = args.rigid_thr
    if 'cardboard' in name2id:
        per_class_thr[name2id['cardboard']] = args.cardboard_thr
    if 'metal' in name2id:
        per_class_thr[name2id['metal']] = args.metal_thr
    if 'soft_plastic' in name2id:
        per_class_thr[name2id['soft_plastic']] = args.soft_thr

    per_class_nms_iou = {cid: args.default_nms_iou for cid in coco_gt.getCatIds()}
    if 'metal' in name2id:
        per_class_nms_iou[name2id['metal']] = args.metal_nms_iou

    filtered_preds = filter_predictions(pred_bucket, per_class_thr, per_class_nms_iou)

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_preds, f)

    print(f'[OK] saved filtered predictions: {out_path}')
    print(f'[OK] num predictions: {len(filtered_preds)}')

    if not args.skip_eval:
        max_dets = [int(x.strip()) for x in args.max_dets.split(',') if x.strip()]
        evaluate(coco_gt, filtered_preds, max_dets=max_dets)


if __name__ == '__main__':
    main()
