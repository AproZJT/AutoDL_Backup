import os
from mmengine.config import Config

base_cfg_path = 'external_modules/mmdetection/configs/grounding_dino/grounding_dino_swin-b_inference_zerowaste_f.py'
cfg = Config.fromfile(base_cfg_path)

# 1. 联合原始训练集和伪标签集
original_train_dataset = cfg.train_dataloader.dataset

pseudo_dataset = cfg.train_dataloader.dataset.copy()
pseudo_dataset.ann_file = 'test/pseudo_labels.json' 

cfg.train_dataloader.dataset = dict(
    type='ConcatDataset',
    datasets=[original_train_dataset, pseudo_dataset]
)

# 2. 冻结策略与保守超参
cfg.model.bbox_head.loss_cls.loss_weight = 1.0 
cfg.model.bbox_head.loss_bbox.loss_weight = 2.0 

cfg.train_cfg.max_epochs = 2 
cfg.train_cfg.val_interval = 1

# 极其保守的学习率 (降维打击)
if 'optim_wrapper' in cfg:
    cfg.optim_wrapper.optimizer.lr = 1e-5 

# 加载最强的 SWA 权重作为起点
cfg.load_from = 'weights/final_sota/best_swa_0.545.pth'
cfg.work_dir = './work_dirs/pseudo_finetune_ultimate'

cfg.dump('pseudo_finetune_cfg.py')
print("✅ 伪标签微调 Config 已生成！")
