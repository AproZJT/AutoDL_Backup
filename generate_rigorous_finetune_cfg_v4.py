from mmengine.config import Config

base_cfg = 'external_modules/mmdetection/configs/grounding_dino/grounding_dino_swin-b_semi-sup_zerowaste_finetune.py'
sota = 'weights/final_sota/best_swa_0.545.pth'

cfg = Config.fromfile(base_cfg)

# 原始训练集保持不动
orig_train = cfg.train_dataloader.dataset

# 显式构建“可训练”的伪标签数据集（别再 copy wrapper）
pseudo_train = dict(
    type='CocoDataset',
    data_root='./data/zerowaste-f/',
    ann_file='unlabeled_pseudo.json',
    data_prefix=dict(img='train/data/'),
    metainfo=dict(classes=('rigid_plastic', 'cardboard', 'metal', 'soft_plastic')),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=cfg.train_dataloader.dataset.pipeline if isinstance(cfg.train_dataloader.dataset, dict) and 'pipeline' in cfg.train_dataloader.dataset else cfg.train_pipeline
)

cfg.train_dataloader.dataset = dict(
    type='ConcatDataset',
    datasets=[orig_train, pseudo_train]
)

cfg.train_cfg.max_epochs = 1
cfg.train_cfg.val_interval = 1
cfg.optim_wrapper.optimizer.lr = 1e-5
cfg.load_from = sota
cfg.work_dir = './work_dirs/rigorous_pseudo_finetune_v4'

cfg.dump('rigorous_finetune_cfg_v4.py')
print('[OK] rigorous_finetune_cfg_v4.py')
