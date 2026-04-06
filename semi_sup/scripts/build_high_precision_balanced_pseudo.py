import json
import argparse
from collections import defaultdict

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="input coco pseudo json")
    p.add_argument("--output", required=True, help="output filtered coco pseudo json")
    p.add_argument("--default_thr", type=float, default=0.80)
    p.add_argument("--per_class_thr_json", type=str, default='{"metal":0.75}')
    p.add_argument("--per_class_topk", type=int, default=2000)
    return p.parse_args()

def main():
    args = parse_args()
    data = json.load(open(args.input, "r", encoding="utf-8"))
    anns = data["annotations"]
    cats = data["categories"]
    id2name = {c["id"]: c["name"] for c in cats}
    per_class_thr = json.loads(args.per_class_thr_json)

    by_cls = defaultdict(list)
    for a in anns:
        cls_name = id2name[a["category_id"]]
        thr = per_class_thr.get(cls_name, args.default_thr)
        if float(a.get("score", 0.0)) >= thr:
            by_cls[a["category_id"]].append(a)

    kept = []
    for cid, arr in by_cls.items():
        arr.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        kept.extend(arr[:args.per_class_topk])

    for i, a in enumerate(kept, 1):
        a["id"] = i
        if "area" not in a:
            x, y, w, h = a["bbox"]
            a["area"] = float(max(w, 0) * max(h, 0))
        if "iscrowd" not in a:
            a["iscrowd"] = 0

    out = {
        "images": data["images"],
        "categories": data["categories"],
        "annotations": kept
    }
    json.dump(out, open(args.output, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"[OK] saved: {args.output}")
    print(f"[OK] kept anns: {len(kept)}")
    for c in cats:
        cnt = sum(1 for a in kept if a["category_id"] == c["id"])
        print(f"  - {c['name']}: {cnt}")

if __name__ == "__main__":
    main()
