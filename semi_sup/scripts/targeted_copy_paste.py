import json
import os
import cv2
import random
import argparse
import copy
from tqdm import tqdm

def bbox_to_int(bbox):
    return [int(max(0, x)) for x in bbox]

def main():
    parser = argparse.ArgumentParser(description="Metal 专属增量 Copy-Paste")
    parser.add_argument('--json_path', required=True, help='原始训练集 JSON')
    parser.add_argument('--img_dir', required=True, help='原始图片所在目录')
    parser.add_argument('--out_json', required=True, help='增强后的新 JSON')
    parser.add_argument('--aug_num', type=int, default=500, help='生成多少张新图片')
    args = parser.parse_args()

    print(f"[INFO] 加载标注文件: {args.json_path}")
    with open(args.json_path, 'r') as f:
        data = json.load(f)

    # 动态获取 ID
    cat_name_to_id = {cat['name']: cat['id'] for cat in data['categories']}
    metal_id = cat_name_to_id.get('metal')
    
    # 建立索引
    img_dict = {img['id']: img for img in data['images']}
    ann_by_img = {img['id']: [] for img in data['images']}
    metal_crops_info = []

    for ann in data['annotations']:
        ann_by_img[ann['image_id']].append(ann)
        if ann['category_id'] == metal_id:
            metal_crops_info.append(ann)

    print(f"[INFO] 提取到 {len(metal_crops_info)} 个高质量 Metal 候选框")
    
    if len(metal_crops_info) == 0:
        print("[ERROR] 找不到 metal 标注，请检查 JSON！")
        return

    new_images = []
    new_annotations = []
    
    # 获取当前最大 ID，防止冲突
    max_img_id = max([img['id'] for img in data['images']]) if data['images'] else 0
    max_ann_id = max([ann['id'] for ann in data['annotations']]) if data['annotations'] else 0

    print(f"[INFO] 正在生成 {args.aug_num} 张 Metal 增强图...")
    for i in tqdm(range(args.aug_num)):
        # 1. 随机选一张背景图
        bg_img_info = random.choice(data['images'])
        bg_path = os.path.join(args.img_dir, bg_img_info['file_name'])
        bg_img = cv2.imread(bg_path)
        if bg_img is None: continue
        
        # 2. 准备新的图片信息
        max_img_id += 1
        new_img_name = f"cp_aug_metal_{max_img_id}.jpg"
        new_img_path = os.path.join(args.img_dir, new_img_name)
        
        # 继承原图的所有标注
        current_anns = copy.deepcopy(ann_by_img[bg_img_info['id']])
        for ann in current_anns:
            max_ann_id += 1
            ann['id'] = max_ann_id
            ann['image_id'] = max_img_id
            new_annotations.append(ann)

        # 3. 随机贴 1~3 个金属块
        num_pastes = random.randint(1, 3)
        for _ in range(num_pastes):
            crop_info = random.choice(metal_crops_info)
            src_img_info = img_dict[crop_info['image_id']]
            src_path = os.path.join(args.img_dir, src_img_info['file_name'])
            src_img = cv2.imread(src_path)
            if src_img is None: continue
            
            x, y, w, h = bbox_to_int(crop_info['bbox'])
            # 越界保护
            if w <= 5 or h <= 5 or x+w > src_img.shape[1] or y+h > src_img.shape[0]: continue
            
            crop_patch = src_img[y:y+h, x:x+w]
            
            # 随机选择粘贴位置 (预留边界)
            bg_h, bg_w = bg_img.shape[:2]
            if bg_w - w <= 0 or bg_h - h <= 0: continue
            
            paste_x = random.randint(0, bg_w - w - 1)
            paste_y = random.randint(0, bg_h - h - 1)
            
            # 直接物理覆盖像素
            bg_img[paste_y:paste_y+h, paste_x:paste_x+w] = crop_patch
            
            # 增加新标注
            max_ann_id += 1
            new_ann = copy.deepcopy(crop_info)
            new_ann['id'] = max_ann_id
            new_ann['image_id'] = max_img_id
            new_ann['bbox'] = [float(paste_x), float(paste_y), float(w), float(h)]
            new_ann['area'] = float(w * h)
            new_annotations.append(new_ann)

        # 保存新图片
        cv2.imwrite(new_img_path, bg_img)
        
        # 登记新图片信息
        new_img_info = copy.deepcopy(bg_img_info)
        new_img_info['id'] = max_img_id
        new_img_info['file_name'] = new_img_name
        new_images.append(new_img_info)

    # 4. 合并数据并保存
    data['images'].extend(new_images)
    data['annotations'].extend(new_annotations)
    
    with open(args.out_json, 'w') as f:
        json.dump(data, f)
        
    print(f"\n[SUCCESS] Copy-Paste 增强完成！")
    print(f"新增图片: {len(new_images)} 张，已保存在原图片目录。")
    print(f"增强后的 JSON 已保存至: {args.out_json}\n")

if __name__ == '__main__':
    main()
