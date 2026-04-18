#!/usr/bin/env python3
import json
import csv
import copy
import argparse
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def iou_xywh(a, b):
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
    return inter / ua if ua > 0 else 0.0

def nms_xywh(dets, iou_thr):
    if not dets:
        return []
    dets = sorted(dets, key=lambda x: x['score'], reverse=True)
    keep = []
    while dets:
        best = dets.pop(0)
        keep.append(best)
        left = []
        for d in dets:
            if iou_xywh(best['bbox'], d['bbox']) <= iou_thr:
                left.append(d)
        dets = left
    return keep

def postprocess(
    preds,
    cat_id_to_name,
    default_thr,
    default_nms_iou,
    metal_thr,
    metal_nms_iou,
    max_dets=1000,
):
    # 1) threshold
    filtered = []
    for p in preds:
        cat_name = cat_id_to_name.get(p['category_id'], '')
        thr = metal_thr if cat_name == 'metal' else default_thr
        if p.get('score', 0.0) >= thr:
            filtered.append(p)

    # 2) group by (image_id, category_id)
    groups = {}
    for p in filtered:
        k = (p['image_id'], p['category_id'])
        groups.setdefault(k, []).append(p)

    # 3) class-aware nms
    out = []
    for (img_id, cat_id), dets in groups.items():
        cat_name = cat_id_to_name.get(cat_id, '')
        iou_thr = metal_nms_iou if cat_name == 'metal' else default_nms_iou
        if iou_thr >= 1.0:
            kept = sorted(dets, key=lambda x: x['score'], reverse=True)
        else:
            kept = nms_xywh(dets, iou_thr)
        out.extend(kept)

    # 4) max dets per image
    if max_dets is not None and max_dets > 0:
        by_img = {}
        for p in out:
            by_img.setdefault(p['image_id'], []).append(p)
        out2 = []
        for _, arr in by_img.items():
            arr = sorted(arr, key=lambda x: x['score'], reverse=True)[:max_dets]
            out2.extend(arr)
        out = out2

    return out

def coco_eval(gt_json, pred_json):
    coco_gt = COCO(gt_json)
    coco_dt = coco_gt.loadRes(pred_json)
    e = COCOeval(coco_gt, coco_dt, 'bbox')
    e.evaluate()
    e.accumulate()
    e.summarize()
    return {
        'mAP': float(e.stats[0]),
        'mAP50': float(e.stats[1]),
        'mAP75': float(e.stats[2]),
        'mAP_s': float(e.stats[3]),
        'mAP_m': float(e.stats[4]),
        'mAP_l': float(e.stats[5]),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_json', required=True)
    parser.add_argument('--pred_json', required=True)
    parser.add_argument('--out_dir', default='./data/pseudo_labels/sweeps')
    parser.add_argument('--default_thr', type=float, default=0.0)
    parser.add_argument('--default_nms_iou', type=float, default=1.0)
    parser.add_argument('--metal_nms_list', default='0.70,0.72,0.74,0.76')
    parser.add_argument('--metal_thr_list', default='0.00,0.01,0.02,0.03')
    parser.add_argument('--max_dets', type=int, default=1000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.gt_json, 'r', encoding='utf-8') as f:
        gt = json.load(f)
    with open(args.pred_json, 'r', encoding='utf-8') as f:
        preds = json.load(f)

    cat_id_to_name = {c['id']: c['name'] for c in gt['categories']}

    # baseline (raw)
    baseline_path = out_dir / 'baseline_raw_copy.bbox.json'
    with open(baseline_path, 'w', encoding='utf-8') as f:
        json.dump(preds, f)
    print('\n=== Baseline RAW ===')
    baseline_metrics = coco_eval(args.gt_json, str(baseline_path))

    metal_nms_list = [float(x) for x in args.metal_nms_list.split(',') if x.strip()]
    metal_thr_list = [float(x) for x in args.metal_thr_list.split(',') if x.strip()]

    rows = []
    best = None

    for metal_nms in metal_nms_list:
        for metal_thr in metal_thr_list:
            post = postprocess(
                preds=copy.deepcopy(preds),
                cat_id_to_name=cat_id_to_name,
                default_thr=args.default_thr,
                default_nms_iou=args.default_nms_iou,
                metal_thr=metal_thr,
                metal_nms_iou=metal_nms,
                max_dets=args.max_dets,
            )

            out_json = out_dir / f'post_mnms_{metal_nms:.2f}_mthr_{metal_thr:.2f}.bbox.json'
            with open(out_json, 'w', encoding='utf-8') as f:
                json.dump(post, f)

            print(f'\n=== Eval metal_nms={metal_nms:.2f}, metal_thr={metal_thr:.2f} ===')
            m = coco_eval(args.gt_json, str(out_json))
            row = {
                'metal_nms_iou': metal_nms,
                'metal_thr': metal_thr,
                'default_thr': args.default_thr,
                'default_nms_iou': args.default_nms_iou,
                'mAP': m['mAP'],
                'mAP50': m['mAP50'],
                'mAP75': m['mAP75'],
                'delta_mAP_vs_raw': m['mAP'] - baseline_metrics['mAP'],
                'pred_json': str(out_json),
                'num_preds': len(post),
            }
            rows.append(row)

            if best is None or row['mAP'] > best['mAP']:
                best = row

    csv_path = out_dir / 'grid_search_report.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda x: x['mAP'], reverse=True))

    best_path = out_dir / 'grid_search_best.json'
    with open(best_path, 'w', encoding='utf-8') as f:
        json.dump({'baseline_raw': baseline_metrics, 'best': best}, f, indent=2)

    print('\n================ FINAL =================')
    print(f"RAW mAP:  {baseline_metrics['mAP']:.6f}")
    print(f"BEST mAP: {best['mAP']:.6f}")
    print(f"DELTA:    {best['delta_mAP_vs_raw']:+.6f}")
    print(f"BEST PARAMS: metal_nms_iou={best['metal_nms_iou']}, metal_thr={best['metal_thr']}")
    print(f"CSV:  {csv_path}")
    print(f"BEST: {best_path}")

if __name__ == '__main__':
    main()
