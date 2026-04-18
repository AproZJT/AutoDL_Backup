from mmengine.config import Config
cfg = Config.fromfile('external_modules/mmdetection/configs/grounding_dino/grounding_dino_swin-b_semi-sup_zerowaste_finetune.py')

orig = cfg.train_dataloader.dataset
pseudo = cfg.train_dataloader.dataset.copy()

inner = pseudo
while isinstance(inner, dict) and 'dataset' in inner:
    inner = inner['dataset']

inner['ann_file'] = 'unlabeled_pseudo.json'
inner['data_prefix'] = dict(img='train/data/')

cfg.train_dataloader.dataset = dict(
    type='ConcatDataset',
    datasets=[orig, pseudo]
)

cfg.train_cfg.max_epochs = 1
cfg.train_cfg.val_interval = 1
if 'optim_wrapper' in cfg and 'optimizer' in cfg.optim_wrapper:
    cfg.optim_wrapper.optimizer.lr = 1e-5

cfg.load_from = 'weights/final_sota/best_swa_0.545.pth'
cfg.work_dir = './work_dirs/rigorous_pseudo_finetune_v3'
cfg.dump('rigorous_finetune_cfg_v3.py')
print('[OK] rigorous_finetune_cfg_v3.py')
