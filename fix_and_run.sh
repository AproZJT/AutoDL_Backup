#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTHONPATH=.

BASE_CFG="external_modules/mmdetection/configs/grounding_dino/grounding_dino_swin-b_inference_zerowaste_f.py"
SOTA_WEIGHT="weights/final_sota/best_swa_0.545.pth"

# ==========================================
# 🚨 核心修复：强制指定无标签图片的真实存放目录
# 假设图片存放在 data/zerowaste-f/unlabeled/data/ 中
# ==========================================
IMG_PREFIX="unlabeled/data/"

echo "=================================================="
echo "🎯 Step 1: 修复 Unlabeled 推理 Config (纠正图片路径)"
echo "=================================================="
cat << PY_EOF > generate_unlabeled_infer_cfg.py
import sys
from mmengine.config import Config

cfg = Config.fromfile(sys.argv[1])
cfg.test_dataloader.dataset.ann_file = 'unlabeled/labels.json'
# 【修复点】：强行覆盖图片前缀，切断对 test 目录的继承！
cfg.test_dataloader.dataset.data_prefix = dict(img='$IMG_PREFIX')

cfg.dump('unlabeled_infer_cfg.py')
PY_EOF
python generate_unlabeled_infer_cfg.py $BASE_CFG

echo "=================================================="
echo "🚀 Step 2: 重新提取 Unlabeled 预测"
echo "=================================================="
python external_modules/mmdetection/tools/test.py unlabeled_infer_cfg.py $SOTA_WEIGHT \
  --cfg-options test_evaluator.outfile_prefix=./data/pseudo_labels/unlabeled_raw

echo "=================================================="
echo "🧹 Step 3: 提取高纯度伪标签"
echo "=================================================="
python semi_sup/scripts/make_pseudo_coco.py \
  --test_gt_json ./data/zerowaste-f/unlabeled/labels.json \
  --pred_json ./data/pseudo_labels/unlabeled_raw.bbox.json \
  --out_json ./data/zerowaste-f/unlabeled_pseudo.json \
  --thr_metal 0.20 \
  --thr_others 0.30

echo "=================================================="
echo "🧬 Step 4: 修复微调 Config (联合训练)"
echo "=================================================="
cat << PY_EOF > generate_rigorous_finetune_cfg.py
import sys
from mmengine.config import Config

cfg = Config.fromfile(sys.argv[1])

original_train_dataset = cfg.train_dataloader.dataset

pseudo_dataset = cfg.train_dataloader.dataset.copy()
# 使用纯相对路径
pseudo_dataset['ann_file'] = 'unlabeled_pseudo.json'
# 【核心修复】：在训练时，伪标签的数据集也必须强制指向正确的图片目录！
pseudo_dataset['data_prefix'] = dict(img='$IMG_PREFIX')

cfg.train_dataloader.dataset = dict(
    type='ConcatDataset',
    datasets=[original_train_dataset, pseudo_dataset]
)

# 冻结与超参设定
cfg.model.bbox_head.loss_cls.loss_weight = 1.0
cfg.model.bbox_head.loss_bbox.loss_weight = 2.0
cfg.train_cfg.max_epochs = 1
cfg.train_cfg.val_interval = 1

if 'optim_wrapper' in cfg:
    cfg.optim_wrapper.optimizer.lr = 1e-5

cfg.load_from = sys.argv[2]
cfg.work_dir = './work_dirs/rigorous_pseudo_finetune'
cfg.dump('rigorous_finetune_cfg.py')
PY_EOF
python generate_rigorous_finetune_cfg.py $BASE_CFG $SOTA_WEIGHT

echo "=================================================="
echo "🔥 Step 5: 重新点火！执行 1 Epoch 终极微调"
echo "=================================================="
python external_modules/mmdetection/tools/train.py rigorous_finetune_cfg.py
