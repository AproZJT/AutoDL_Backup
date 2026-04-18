#!/usr/bin/env bash
set -euo pipefail

# =========================
# 0) 环境准备
# =========================
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTHONPATH=.

PROJECT_ROOT="$(pwd)"
DATA_ROOT="./data/zerowaste-f"
RAW_PREFIX="train/data/"
UNLABELED_JSON="$DATA_ROOT/unlabeled/labels.json"
SOTA_WEIGHT="weights/final_sota/best_swa_0.545.pth"
TRAIN_BASE_CFG="external_modules/mmdetection/configs/grounding_dino/grounding_dino_swin-b_semi-sup_zerowaste.py"

mkdir -p ./data/pseudo_labels ./semi_sup/scripts

echo "=================================================="
echo "[1/4] 生成 Unlabeled 推理配置（修复 evaluator 对齐）"
echo "=================================================="
cat > generate_unlabeled_infer_cfg.py << 'PY_EOF'
from copy import deepcopy
from mmengine.config import Config

cfg = Config.fromfile('external_modules/mmdetection/configs/grounding_dino/grounding_dino_swin-b_semi-sup_zerowaste.py')

unlabeled_ann = 'zerowaste-f/unlabeled/labels.json'
raw_prefix = 'train/data/'

def set_ann_and_prefix(ds):
    if isinstance(ds, dict) and 'dataset' in ds:
        set_ann_and_prefix(ds['dataset'])
        return
    if isinstance(ds, dict) and 'datasets' in ds:
        for item in ds['datasets']:
            set_ann_and_prefix(item)
        return
    if isinstance(ds, dict):
        ds['ann_file'] = unlabeled_ann
        ds['data_prefix'] = dict(img=raw_prefix)
        ds['filter_cfg'] = dict(filter_empty_gt=False, min_size=0)

cfg.test_dataloader = deepcopy(cfg.val_dataloader)
set_ann_and_prefix(cfg.test_dataloader['dataset'])

# 关键修复：evaluator ann_file 必须和 test_dataloader 一致
if 'test_evaluator' not in cfg or cfg.test_evaluator is None:
    cfg.test_evaluator = dict(type='CocoMetric', metric='bbox', ann_file='./data/' + unlabeled_ann)
else:
    cfg.test_evaluator['ann_file'] = './data/' + unlabeled_ann

cfg.val_evaluator = cfg.test_evaluator
cfg.dump('unlabeled_infer_cfg.py')
print('Generated unlabeled_infer_cfg.py')
PY_EOF
python generate_unlabeled_infer_cfg.py

echo "=================================================="
echo "[2/4] 执行推理并导出 bbox json"
echo "=================================================="
python external_modules/mmdetection/tools/test.py \
    unlabeled_infer_cfg.py "$SOTA_WEIGHT" \
    --cfg-options test_evaluator.outfile_prefix=./data/pseudo_labels/unlabeled_raw

echo "=================================================="
echo "[3/4] 提纯伪标签并生成联合训练配置"
echo "=================================================="
cat > semi_sup/scripts/make_pseudo_coco.py << 'PY_EOF'
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

    cat_id_to_name = {c['id']: c['name'] for c in coco['categories']}
    valid_img_ids = {img['id'] for img in coco['images']}

    pseudo_anns = []
    ann_id = 1
    skipped_mismatch = 0

    for p in preds:
        if p['image_id'] not in valid_img_ids:
            skipped_mismatch += 1
            continue

        cat_name = cat_id_to_name[p['category_id']]
        thr = args.thr_metal if cat_name == 'metal' else args.thr_others

        if p['score'] >= thr:
            x, y, w, h = p['bbox']
            pseudo_anns.append({
                'id': ann_id,
                'image_id': p['image_id'],
                'category_id': p['category_id'],
                'bbox': [x, y, w, h],
                'area': w * h,
                'iscrowd': 0,
                'score': p['score'],
            })
            ann_id += 1

    coco['annotations'] = pseudo_anns
    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump(coco, f)

    print(f'成功提取 {len(pseudo_anns)} 个高质量伪标签 -> {args.out_json}')
    if skipped_mismatch > 0:
        print(f'警告: 跳过 {skipped_mismatch} 个 image_id 不在 unlabeled GT 的预测框')

if __name__ == '__main__':
    main()
PY_EOF

python semi_sup/scripts/make_pseudo_coco.py \
    --test_gt_json "$UNLABELED_JSON" \
    --pred_json ./data/pseudo_labels/unlabeled_raw.bbox.json \
    --out_json "$DATA_ROOT/unlabeled_pseudo.json"

cat > generate_rigorous_finetune_cfg_v3.py << 'PY_EOF'
from copy import deepcopy
from mmengine.config import Config

cfg = Config.fromfile('external_modules/mmdetection/configs/grounding_dino/grounding_dino_swin-b_semi-sup_zerowaste.py')

original_train_dataset = deepcopy(cfg.train_dataloader.dataset)
pseudo_dataset = deepcopy(cfg.train_dataloader.dataset)

def find_first_leaf_coco(ds):
    if isinstance(ds, dict) and 'dataset' in ds:
        return find_first_leaf_coco(ds['dataset'])
    if isinstance(ds, dict) and 'datasets' in ds and len(ds['datasets']) > 0:
        return find_first_leaf_coco(ds['datasets'][0])
    return ds

leaf = find_first_leaf_coco(pseudo_dataset)
leaf['ann_file'] = './data/zerowaste-f/unlabeled_pseudo.json'
leaf['data_prefix'] = dict(img='./data/zerowaste-f/train/data')

cfg.train_dataloader.dataset = dict(
    type='ConcatDataset',
    datasets=[original_train_dataset, pseudo_dataset],
)

cfg.train_cfg.max_epochs = 1
cfg.train_cfg.val_interval = 1
if 'optim_wrapper' in cfg and 'optimizer' in cfg.optim_wrapper:
    cfg.optim_wrapper.optimizer.lr = 1e-5

cfg.load_from = 'weights/final_sota/best_swa_0.545.pth'
cfg.work_dir = './work_dirs/rigorous_pseudo_finetune_v3'
cfg.dump('rigorous_finetune_cfg_v3.py')
print('Generated rigorous_finetune_cfg_v3.py')
PY_EOF
python generate_rigorous_finetune_cfg_v3.py

echo "=================================================="
echo "[4/4] 正式训练（1 epoch 自蒸馏）"
echo "=================================================="
python external_modules/mmdetection/tools/train.py rigorous_finetune_cfg_v3.py

echo "全部完成。请在 ./work_dirs/rigorous_pseudo_finetune_v3 查看结果。"
