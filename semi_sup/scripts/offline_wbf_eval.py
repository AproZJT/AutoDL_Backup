import json
import argparse
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ensemble_boxes import weighted_boxes_fusion

def convert_to_wbf(coco_dt, img_w, img_h):
    boxes, scores, labels = [], [], []
    for ann in coco_dt:
        x, y, w, h = ann['bbox']
        boxes.append([
            max(0.0, x / img_w), 
            max(0.0, y / img_h), 
            min(1.0, (x + w) / img_w), 
            min(1.0, (y + h) / img_h)
        ])
        scores.append(ann['score'])
        labels.append(ann['category_id'])
    return boxes, scores, labels

def convert_to_coco(boxes, scores, labels, img_id, img_w, img_h):
    anns = []
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        anns.append({
            'image_id': img_id,
            'category_id': int(label),
            'bbox': [x1 * img_w, y1 * img_h, (x2 - x1) * img_w, (y2 - y1) * img_h],
            'score': float(score)
        })
    return anns

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_json', required=True)
    parser.add_argument('--pred_jsons', nargs='+', required=True)
    parser.add_argument('--wbf_iou_thr', type=float, default=0.60)
    parser.add_argument('--skip_box_thr', type=float, default=0.0001)
    args = parser.parse_args()

    coco_gt = COCO(args.gt_json)
    img_dict = {img['id']: img for img in coco_gt.dataset['images']}

    preds_by_model = []
    for p in args.pred_jsons:
        with open(p, 'r') as f:
            preds_by_model.append(json.load(f))

    img_preds = {img_id: [[] for _ in args.pred_jsons] for img_id in img_dict}
    for m_idx, preds in enumerate(preds_by_model):
        for p in preds:
            if p['image_id'] in img_preds:
                img_preds[p['image_id']][m_idx].append(p)

    fused_annotations = []
    weights = [1.0] * len(args.pred_jsons)

    for img_id, model_anns in img_preds.items():
        img_w, img_h = img_dict[img_id]['width'], img_dict[img_id]['height']
        b_list, s_list, l_list = [], [], []
        
        for anns in model_anns:
            b, s, l = convert_to_wbf(anns, img_w, img_h)
            b_list.append(b)
            s_list.append(s)
            l_list.append(l)

        if sum(len(b) for b in b_list) > 0:
            fb, fs, fl = weighted_boxes_fusion(
                b_list, s_list, l_list, weights=weights, 
                iou_thr=args.wbf_iou_thr, skip_box_thr=args.skip_box_thr
            )
            fused_annotations.extend(convert_to_coco(fb, fs, fl, img_id, img_w, img_h))

    coco_dt = coco_gt.loadRes(fused_annotations)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.params.maxDets = [100, 300, 1000]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize() # <--- 致命的缺漏已补上！
    
    print(f"\n🔥 WBF Result | IoU Thr: {args.wbf_iou_thr} | mAP: {coco_eval.stats[0]:.6f}")

if __name__ == '__main__':
    main()
