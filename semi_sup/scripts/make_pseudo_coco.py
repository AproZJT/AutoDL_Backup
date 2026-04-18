import json
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_gt_json', required=True)
    parser.add_argument('--pred_json', required=True)
    parser.add_argument('--out_json', required=True)
    parser.add_argument('--thr_metal', type=float, default=0.20)
    parser.add_argument('--thr_others', type=float, default=0.30)
    args = parser.parse_args()

    with open(args.test_gt_json, 'r', encoding='utf-8') as f:
        coco = json.load(f)
    with open(args.pred_json, 'r', encoding='utf-8') as f:
        preds = json.load(f)

    cat_id_to_name = {c['id']: c['name'] for c in coco.get('categories', [])}
    pseudo_anns = []
    ann_id = 1
    filtered = 0
    kept_cls = {}

    for p in preds:
        cid = p['category_id']
        cname = cat_id_to_name.get(cid, '')
        thr = args.thr_metal if cname == 'metal' else args.thr_others
        if p.get('score', 0.0) >= thr:
            x, y, w, h = p['bbox']
            pseudo_anns.append({
                "id": ann_id,
                "image_id": p['image_id'],
                "category_id": cid,
                "bbox": [x, y, w, h],
                "area": float(w * h),
                "iscrowd": 0
            })
            kept_cls[cname] = kept_cls.get(cname, 0) + 1
            ann_id += 1
        else:
            filtered += 1

    coco['annotations'] = pseudo_anns
    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump(coco, f)

    print("===== 正在提取高纯度伪标签 =====")
    print(f"[OK] 过滤掉的低分噪点数: {filtered}")
    print("[OK] 成功提取的高质量伪标签:")
    for k,v in kept_cls.items():
        print(f"  - {k}: {v} 个")
    print(f"[OK] 伪标签集已保存至: {args.out_json}")

if __name__ == '__main__':
    main()
