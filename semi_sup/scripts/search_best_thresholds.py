import argparse
import csv
import itertools
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def parse_float_list(v: str) -> List[float]:
    return [float(x.strip()) for x in v.split(',') if x.strip()]


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


def build_cat_maps(coco_gt: COCO) -> Tuple[Dict[str, int], Dict[int, str]]:
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


def eval_coco(coco_gt: COCO, filtered_preds: List[dict], max_dets: List[int]) -> Tuple[float, Dict[int, float]]:
    if len(filtered_preds) == 0:
        return 0.0, {}

    coco_dt = coco_gt.loadRes(filtered_preds)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.params.maxDets = max_dets
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    overall_map = float(coco_eval.stats[0])

    precision = coco_eval.eval['precision']
    cat_ids = coco_gt.getCatIds()
    per_class_ap = {}

    for k, cat_id in enumerate(cat_ids):
        p = precision[:, :, k, 0, -1]
        p = p[p > -1]
        per_class_ap[int(cat_id)] = float(np.mean(p)) if p.size else 0.0

    return overall_map, per_class_ap


def parse_args():
    parser = argparse.ArgumentParser(description='Offline threshold/NMS search for COCO predictions')
    parser.add_argument('--gt_json', type=str, required=True)
    parser.add_argument('--pred_json', type=str, required=True)
    parser.add_argument('--out_csv', type=str, default='data/pseudo_labels/sweeps/threshold_search_results.csv')

    parser.add_argument('--target_class', type=str, default='metal')
    parser.add_argument('--target_thr_grid', type=str, default='0.00,0.01,0.02,0.03,0.05')

    parser.add_argument('--default_thr', type=float, default=0.0)
    parser.add_argument('--search_all_classes', action='store_true')
    parser.add_argument('--rigid_grid', type=str, default='0.00')
    parser.add_argument('--cardboard_grid', type=str, default='0.00')
    parser.add_argument('--metal_grid', type=str, default='0.00')
    parser.add_argument('--soft_grid', type=str, default='0.00')

    parser.add_argument('--enable_nms_search', action='store_true')
    parser.add_argument('--default_nms_iou', type=float, default=1.0)
    parser.add_argument('--target_nms_grid', type=str, default='0.72')

    # New: class-wise NMS grids
    parser.add_argument('--search_all_class_nms', action='store_true', help='Enable cartesian search for per-class NMS IoU')
    parser.add_argument('--rigid_nms_grid', type=str, default='1.00')
    parser.add_argument('--cardboard_nms_grid', type=str, default='1.00')
    parser.add_argument('--metal_nms_grid', type=str, default='0.72')
    parser.add_argument('--soft_nms_grid', type=str, default='1.00')

    parser.add_argument('--max_dets', type=str, default='100,300,1000')
    parser.add_argument('--topk', type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()

    gt_path = Path(args.gt_json)
    pred_path = Path(args.pred_json)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    coco_gt = COCO(str(gt_path))
    name2id, _ = build_cat_maps(coco_gt)

    with open(pred_path, 'r', encoding='utf-8') as f:
        raw_preds = json.load(f)

    pred_bucket = bucket_predictions(raw_preds)

    required = ['rigid_plastic', 'cardboard', 'metal', 'soft_plastic']
    missing = [n for n in required if n not in name2id]
    if missing:
        raise ValueError(f'Missing categories in GT: {missing}')

    class_ids = [name2id[n] for n in required]
    target_cat_id = name2id[args.target_class]

    target_thr_list = parse_float_list(args.target_thr_grid)
    target_nms_list = parse_float_list(args.target_nms_grid) if args.enable_nms_search else [args.default_nms_iou]

    if args.search_all_class_nms:
        rigid_nms_list = parse_float_list(args.rigid_nms_grid)
        cardboard_nms_list = parse_float_list(args.cardboard_nms_grid)
        metal_nms_list = parse_float_list(args.metal_nms_grid)
        soft_nms_list = parse_float_list(args.soft_nms_grid)
    else:
        rigid_nms_list = [args.default_nms_iou]
        cardboard_nms_list = [args.default_nms_iou]
        metal_nms_list = target_nms_list
        soft_nms_list = [args.default_nms_iou]

    combinations = []
    if args.search_all_classes:
        rigid_thr_list = parse_float_list(args.rigid_grid)
        card_thr_list = parse_float_list(args.cardboard_grid)
        metal_thr_list = parse_float_list(args.metal_grid)
        soft_thr_list = parse_float_list(args.soft_grid)
    else:
        rigid_thr_list = [args.default_thr]
        card_thr_list = [args.default_thr]
        metal_thr_list = target_thr_list
        soft_thr_list = [args.default_thr]

    for r_t, c_t, m_t, s_t, r_n, c_n, m_n, s_n in itertools.product(
        rigid_thr_list, card_thr_list, metal_thr_list, soft_thr_list,
        rigid_nms_list, cardboard_nms_list, metal_nms_list, soft_nms_list
    ):
        combinations.append((r_t, c_t, m_t, s_t, r_n, c_n, m_n, s_n))

    max_dets = [int(x.strip()) for x in args.max_dets.split(',') if x.strip()]

    rows = []
    start = time.time()

    for idx, (rigid_thr, card_thr, metal_thr, soft_thr, rigid_nms, card_nms, metal_nms, soft_nms) in enumerate(combinations, start=1):
        per_class_thr = {cid: args.default_thr for cid in class_ids}
        per_class_thr[name2id['rigid_plastic']] = rigid_thr
        per_class_thr[name2id['cardboard']] = card_thr
        per_class_thr[name2id['metal']] = metal_thr
        per_class_thr[name2id['soft_plastic']] = soft_thr

        per_class_nms_iou = {cid: args.default_nms_iou for cid in class_ids}
        per_class_nms_iou[name2id['rigid_plastic']] = rigid_nms
        per_class_nms_iou[name2id['cardboard']] = card_nms
        per_class_nms_iou[name2id['metal']] = metal_nms
        per_class_nms_iou[name2id['soft_plastic']] = soft_nms

        filtered_preds = filter_predictions(pred_bucket, per_class_thr, per_class_nms_iou)
        overall_map, per_class_ap = eval_coco(coco_gt, filtered_preds, max_dets=max_dets)
        metal_ap = per_class_ap.get(target_cat_id, 0.0)

        row = {
            'run_idx': idx,
            'rigid_thr': rigid_thr,
            'cardboard_thr': card_thr,
            'metal_thr': metal_thr,
            'soft_thr': soft_thr,
            'rigid_nms_iou': rigid_nms,
            'cardboard_nms_iou': card_nms,
            'metal_nms_iou': metal_nms,
            'soft_nms_iou': soft_nms,
            'num_preds': len(filtered_preds),
            'bbox_mAP': round(overall_map, 6),
            'metal_AP': round(metal_ap, 6),
        }
        rows.append(row)

        if idx % 10 == 0 or idx == len(combinations):
            elapsed = time.time() - start
            best = max(r['bbox_mAP'] for r in rows)
            print(f'[PROGRESS] {idx}/{len(combinations)} | elapsed={elapsed:.1f}s | best={best:.6f}')

    rows_sorted = sorted(rows, key=lambda x: (x['bbox_mAP'], x['metal_AP']), reverse=True)

    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_sorted[0].keys()))
        writer.writeheader()
        writer.writerows(rows_sorted)

    print(f'\n[OK] Saved: {out_csv}')
    print(f'[OK] Top-{args.topk} candidates:')
    for i, r in enumerate(rows_sorted[:args.topk], start=1):
        print(
            f"#{i:02d} mAP={r['bbox_mAP']:.6f} metal={r['metal_AP']:.6f} "
            f"thr=({r['rigid_thr']:.2f},{r['cardboard_thr']:.2f},{r['metal_thr']:.2f},{r['soft_thr']:.2f}) "
            f"nms=({r['rigid_nms_iou']:.2f},{r['cardboard_nms_iou']:.2f},{r['metal_nms_iou']:.2f},{r['soft_nms_iou']:.2f}) "
            f"preds={r['num_preds']}"
        )


if __name__ == '__main__':
    main()
