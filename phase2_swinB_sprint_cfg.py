from mmengine.config import Config
from copy import deepcopy
import os

abs_work_dir = os.getcwd()
BASE_CFG = 'external_modules/mmdetection/configs/grounding_dino/grounding_dino_swin-b_inference_zerowaste_f.py'
cfg = Config.fromfile(BASE_CFG)

# --- 1. 数据集混合 ---
classes = ('rigid_plastic', 'cardboard', 'metal', 'soft_plastic')
metainfo = dict(classes=classes)

def set_ds_info(ds, ann, img_prefix):
    if 'dataset' in ds:
        set_ds_info(ds['dataset'], ann, img_prefix)
        return
    ds.type = 'CocoDataset'
    ds.ann_file = os.path.join(abs_work_dir, 'data', ann)
    ds.data_prefix = dict(img=os.path.join(abs_work_dir, 'data', img_prefix))
    ds.metainfo = metainfo
    ds.data_root = None 
    if 'filter_cfg' in ds:
        ds.filter_cfg = dict(filter_empty_gt=False, min_size=0)

set_ds_info(cfg.train_dataloader.dataset, 'zerowaste-f/train/labels.json', 'zerowaste-f/train/data/')
original_train_ds = deepcopy(cfg.train_dataloader.dataset)

pseudo_dataset = deepcopy(cfg.train_dataloader.dataset)
set_ds_info(pseudo_dataset, 'zerowaste-f/teacher_pseudo_labels_high_prec.json', 'zerowaste-f/train/data/')

cfg.train_dataloader.dataset = dict(
    type='ConcatDataset',
    datasets=[original_train_ds, pseudo_dataset]
)

# --- 2. 彻底移除报错的采样器，回归默认 ---
# ConcatDataset 不支持 ClassAwareSampler，我们换回最稳的 DefaultSampler
cfg.train_dataloader.sampler = dict(type='DefaultSampler', shuffle=True)

# --- 3. 48G 显存满血训练配置 ---
cfg.train_dataloader.batch_size = 4
cfg.optim_wrapper.accumulative_counts = 4
cfg.train_dataloader.num_workers = 4

# 同步修正验证集
set_ds_info(cfg.val_dataloader.dataset, 'zerowaste-f/val/labels.json', 'zerowaste-f/val/data/')
cfg.val_evaluator.ann_file = os.path.join(abs_work_dir, 'data/zerowaste-f/val/labels.json')

# 由于少了均衡采样，我们把 Epoch 拉长一点，给模型更多时间收敛
cfg.train_cfg.max_epochs = 16
cfg.train_cfg.val_interval = 1
cfg.work_dir = './work_dirs/phase2_swinB_sprint'

cfg.dump('phase2_swinB_sprint_final.py')
print("✅ 配置已精简：移除了不兼容的 Sampler，准备开启 Swin-B 满血冲刺！")
