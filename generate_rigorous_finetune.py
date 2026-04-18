from copy import deepcopy
from mmengine.config import Config
import os, json

# 用仓库中存在的配置文件
cfg = Config.fromfile('external_modules/mmdetection/configs/grounding_dino/grounding_dino_swin-b_semi-sup_zerowaste.py')

pseudo_ann_rel = 'zerowaste-f/unlabeled_pseudo.json'
pseudo_ann_abs = './data/' + pseudo_ann_rel

with open(pseudo_ann_abs, 'r', encoding='utf-8') as f:
    coco = json.load(f)

if len(coco.get('images', [])) == 0:
    raise RuntimeError('unlabeled_pseudo.json 里没有 images，无法继续训练。')

sample_name = coco['images'][0]['file_name']
candidates = [
    'zerowaste-f/unlabeled/data/',
    'zerowaste-f/train/data/',
    'unlabeled/data/',
    'train/data/',
    ''
]

picked = None
for p in candidates:
    if os.path.exists(os.path.join('./data', p, sample_name)):
        picked = p
        break

if picked is None:
    raise FileNotFoundError(f'无法定位伪标签图片前缀，sample={sample_name}')

print(f'[AutoDetect] pseudo data_prefix = {picked}')

original_train_dataset = deepcopy(cfg.train_dataloader.dataset)
pseudo_dataset = deepcopy(cfg.train_dataloader.dataset)

def set_ann_and_prefix(ds):
    if isinstance(ds, dict) and 'dataset' in ds:
        set_ann_and_prefix(ds['dataset']); return
    if isinstance(ds, dict) and 'datasets' in ds:
        # 只改其中第一个叶子，避免把原有真标注和伪标注都改坏
        set_ann_and_prefix(ds['datasets'][0]); return
    if isinstance(ds, dict):
        ds['ann_file'] = pseudo_ann_abs
        ds['data_prefix'] = dict(img='./data/' + picked)

set_ann_and_prefix(pseudo_dataset)

cfg.train_dataloader.dataset = dict(
    type='ConcatDataset',
    datasets=[original_train_dataset, pseudo_dataset]
)

# 防 OOM & 微调策略
cfg.train_dataloader.num_workers = 0
cfg.train_dataloader.persistent_workers = False
cfg.train_dataloader.batch_size = 1

if 'optim_wrapper' in cfg and 'optimizer' in cfg.optim_wrapper:
    cfg.optim_wrapper.optimizer.lr = 1e-5

cfg.train_cfg.max_epochs = 1
cfg.train_cfg.val_interval = 1

cfg.load_from = 'weights/final_sota/best_swa_0.545.pth'
cfg.work_dir = './work_dirs/rigorous_pseudo_finetune_v4'
cfg.dump('rigorous_finetune_cfg_v4.py')
print('Generated rigorous_finetune_cfg_v4.py')
