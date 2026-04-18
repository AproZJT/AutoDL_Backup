import sys
from mmengine.config import Config

def patch_dataset(ds, ann_file, img_prefix):
    ds = ds.copy()
    if 'dataset' in ds:
        ds['dataset'] = patch_dataset(ds['dataset'], ann_file, img_prefix)
        return ds
    if 'datasets' in ds:
        ds['datasets'] = [patch_dataset(x, ann_file, img_prefix) for x in ds['datasets']]
        return ds
    ds['ann_file'] = ann_file
    ds['data_prefix'] = dict(img=img_prefix)
    return ds

base_cfg, load_from = sys.argv[1], sys.argv[2]
cfg = Config.fromfile(base_cfg)

orig = cfg.train_dataloader.dataset
pseudo = patch_dataset(orig, 'unlabeled_pseudo.json', 'train/data/')

cfg.train_dataloader.dataset = dict(type='ConcatDataset', datasets=[orig, pseudo])

cfg.train_cfg.max_epochs = 1
cfg.train_cfg.val_interval = 1
cfg.optim_wrapper.optimizer.lr = 1e-5
cfg.load_from = load_from
cfg.work_dir = './work_dirs/rigorous_pseudo_finetune_v2'

cfg.dump('rigorous_finetune_cfg_v2.py')
print('OK: rigorous_finetune_cfg_v2.py')
