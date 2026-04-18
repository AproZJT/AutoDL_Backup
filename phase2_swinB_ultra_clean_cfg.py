from mmengine.config import Config
import os

abs_work_dir = os.getcwd()
BASE_CFG = 'external_modules/mmdetection/configs/grounding_dino/grounding_dino_swin-b_inference_zerowaste_f.py'
cfg = Config.fromfile(BASE_CFG)

classes = ('rigid_plastic', 'cardboard', 'metal', 'soft_plastic')
metainfo = dict(classes=classes)

# 1. 挂载极其纯净的缝合数据集
cfg.train_dataloader.dataset.type = 'CocoDataset'
cfg.train_dataloader.dataset.ann_file = os.path.join(abs_work_dir, 'data/zerowaste-f/merged_train_labels_ultra_clean.json')
cfg.train_dataloader.dataset.data_prefix = dict(img=os.path.join(abs_work_dir, 'data/zerowaste-f/train/data/'))
cfg.train_dataloader.dataset.metainfo = metainfo
cfg.train_dataloader.dataset.data_root = None 
if 'filter_cfg' in cfg.train_dataloader.dataset:
    cfg.train_dataloader.dataset.filter_cfg = dict(filter_empty_gt=False, min_size=0)

cfg.train_dataloader.sampler = dict(type='ClassAwareSampler')
cfg.train_dataloader.batch_size = 4
cfg.train_dataloader.num_workers = 4

cfg.val_dataloader.dataset.type = 'CocoDataset'
cfg.val_dataloader.dataset.ann_file = os.path.join(abs_work_dir, 'data/zerowaste-f/val/labels.json')
cfg.val_dataloader.dataset.data_prefix = dict(img=os.path.join(abs_work_dir, 'data/zerowaste-f/val/data/'))
cfg.val_dataloader.dataset.metainfo = metainfo
cfg.val_dataloader.dataset.data_root = None 
cfg.val_evaluator.ann_file = os.path.join(abs_work_dir, 'data/zerowaste-f/val/labels.json')

# 2. 注入 AdamW 与调度器
cfg.optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.), 'backbone': dict(lr_mult=0.1)}),
    accumulative_counts=4
)

cfg.train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=16, val_interval=1)
cfg.val_cfg = dict(type='ValLoop')

cfg.param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(type='MultiStepLR', begin=0, end=16, by_epoch=True, milestones=[11, 14], gamma=0.1)
]

# ⚠️ 注意：移除了冲突的 EMAHook，保持最纯粹的训练环境
cfg.default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3, save_best='coco/bbox_mAP', rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook')
)

cfg.work_dir = './work_dirs/phase2_swinB_ultra_clean'
cfg.dump('phase2_swinB_ultra_clean.py')
print("✅ 终极 Config (无EMA安全版) 生成完毕！")
