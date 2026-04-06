import json
import torch
from torchvision.ops import nms
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from collections import defaultdict

# ================= 1. 核心路径配置 =================
gt_path = 'data/zerowaste-f/test/labels.json'
pred_path = 'sota_results.bbox.json'  # 刚才导出的 0.545 满血预测文件

print(f"\n[INFO] 🎯 加载官方真实标签: {gt_path}")
coco = COCO(gt_path)
cats = coco.loadCats(coco.getCatIds())
name2id = {c['name']: c['id'] for c in cats}

# ================= 2. 物理属性定制 IoU 阈值 =================
THRESHOLDS = {
    name2id['rigid_plastic']: 0.45,  # 硬塑料：立体排斥，强力去重重复框
    name2id['cardboard']: 0.80,      # 纸板：扁平易堆叠，宽容对待真实堆叠
    name2id['metal']: 0.55,          # 金属：常规阈值
    name2id['soft_plastic']: 0.65    # 软塑料：允许挤压堆叠
}

def main():
    print(f"[INFO] 🚀 加载 0.545 满血预测数据: {pred_path}")
    with open(pred_path, 'r') as f:
        preds = json.load(f)

    # 将预测框按 image_id 和 category_id 分组
    grouped = defaultdict(lambda: defaultdict(list))
    for p in preds:
        grouped[p['image_id']][p['category_id']].append(p)

    final_preds = []
    print("\n🔍 正在进行纯离线物理后处理 (类感知 NMS)...")
    
    # 遍历每张图片的每个类别
    for img_id, cat_dict in grouped.items():
        for cat_id, boxes_list in cat_dict.items():
            if not boxes_list:
                continue

            # 转换为 PyTorch 张量进行 NMS
            boxes, scores = [], []
            for b in boxes_list:
                x, y, w, h = b['bbox']
                # NMS 需要的格式是 [x1, y1, x2, y2]
                boxes.append([x, y, x + w, y + h])
                scores.append(b['score'])

            boxes = torch.tensor(boxes, dtype=torch.float32)
            scores = torch.tensor(scores, dtype=torch.float32)

            # 获取该类别的定制阈值，执行 NMS
            thr = THRESHOLDS.get(cat_id, 0.55)
            keep_idx = nms(boxes, scores, thr)

            # 把幸存下来的框存入最终列表
            for idx in keep_idx:
                final_preds.append(boxes_list[idx.item()])

    # 保存清洗后的结果
    out_json = "sota_filtered_results.json"
    with open(out_json, 'w') as f:
        json.dump(final_preds, f)

    print("\n📊 正在生成最终的过滤成绩单...")
    coco_dt = coco.loadRes(out_json)
    coco_eval = COCOeval(coco, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

if __name__ == "__main__":
    main()