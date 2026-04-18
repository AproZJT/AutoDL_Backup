import json
import argparse
from collections import defaultdict

def is_stable_metal_box(bbox, score, min_score=0.65, min_area=400, max_aspect_ratio=4.0):
    """Metal 专属几何规则过滤 (COCO bbox 格式: [x, y, width, height])"""
    if score < min_score:
        return False
        
    width, height = bbox[2], bbox[3]
    area = width * height
    
    if area < min_area:
        return False
        
    aspect_ratio = max(width / height, height / width) if min(width, height) > 0 else float('inf')
    if aspect_ratio > max_aspect_ratio:
        return False
        
    return True

def main():
    parser = argparse.ArgumentParser(description="安全增强版伪标签提纯 (动态ID + 几何过滤 + TopK)")
    parser.add_argument('--input', required=True, help='初始伪标签 JSON')
    parser.add_argument('--output', required=True, help='提纯后的 JSON')
    parser.add_argument('--metal_score', type=float, default=0.65, help='Metal的基准分数门槛')
    args = parser.parse_args()

    print(f"[INFO] 正在加载伪标签: {args.input}")
    with open(args.input, 'r') as f:
        data = json.load(f)

    # 1. 动态获取类别 ID 映射，彻底避开映射坑
    cat_name_to_id = {cat['name']: cat['id'] for cat in data['categories']}
    cat_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}
    
    metal_id = cat_name_to_id.get('metal')
    if metal_id is None:
        raise ValueError("严重错误：在 categories 中找不到 'metal' 类别！")

    # 预设每类 TopK 上限 (防止类别淹没)
    topk_limits = {
        'rigid_plastic': 1200,
        'cardboard': 2000,
        'metal': 1200,
        'soft_plastic': 2000
    }

    # 按类别暂存 annotations
    cat_anns = defaultdict(list)
    OTHER_CAT_SCORE_THR = 0.80

    for ann in data['annotations']:
        cat_id = ann['category_id']
        score = ann.get('score', 1.0)
        
        if cat_id == metal_id:
            # Metal：分数 + 面积 + 形状 联合过滤
            if is_stable_metal_box(ann['bbox'], score, min_score=args.metal_score):
                cat_anns[cat_id].append(ann)
        else:
            # 非 Metal：死守高置信度门槛
            if score >= OTHER_CAT_SCORE_THR:
                cat_anns[cat_id].append(ann)

    # 2. 排序并执行 TopK 截断
    final_annotations = []
    print("\n[RESULT] 类别配额及过滤结果:")
    for cat_id, anns in cat_anns.items():
        cat_name = cat_id_to_name[cat_id]
        limit = topk_limits.get(cat_name, 1000)
        
        # 按分数从高到低排序
        anns.sort(key=lambda x: x.get('score', 1.0), reverse=True)
        kept_anns = anns[:limit]
        final_annotations.extend(kept_anns)
        
        print(f"         -> {cat_name.ljust(15)}: 候选 {len(anns):>4} | 最终保留 {len(kept_anns):>4} (上限 {limit})")

    data['annotations'] = final_annotations

    with open(args.output, 'w') as f:
        json.dump(data, f)
    print(f"\n[SUCCESS] 高纯度伪标签已保存至: {args.output}")

if __name__ == '__main__':
    main()
