from copy import deepcopy
from mmengine.config import Config
import json
import os

cfg = Config.fromfile('external_modules/mmdetection/configs/grounding_dino/grounding_dino_swin-b_semi-sup_zerowaste.py')

unlabeled_ann_rel = 'zerowaste-f/unlabeled/labels.json'
unlabeled_ann_abs = './data/' + unlabeled_ann_rel

with open(unlabeled_ann_abs, 'r', encoding='utf-8') as f:
    coco = json.load(f)

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
    test_path = os.path.join('./data', p, sample_name)
    if os.path.exists(test_path):
        picked = p
        break

if picked is None:
    raise FileNotFoundError(
        f'无法为样本 {sample_name} 找到有效 data_prefix，请检查 ./data 下真实目录结构。'
    )

print(f'[AutoDetect] sample={sample_name}, data_prefix={picked}')

def set_ann_and_prefix(ds):
    if isinstance(ds, dict) and 'dataset' in ds:
        set_ann_and_prefix(ds['dataset'])
        return
    if isinstance(ds, dict) and 'datasets' in ds:
        for item in ds['datasets']:
            set_ann_and_prefix(item)
        return
    if isinstance(ds, dict):
        ds['ann_file'] = unlabeled_ann_rel
        ds['data_prefix'] = dict(img=picked)
        ds['filter_cfg'] = dict(filter_empty_gt=False, min_size=0)

cfg.test_dataloader = deepcopy(cfg.val_dataloader)
set_ann_and_prefix(cfg.test_dataloader['dataset'])

if 'test_evaluator' not in cfg or cfg.test_evaluator is None:
    cfg.test_evaluator = dict(type='CocoMetric', metric='bbox', ann_file=unlabeled_ann_abs)
else:
    cfg.test_evaluator['ann_file'] = unlabeled_ann_abs

cfg.val_evaluator = cfg.test_evaluator
cfg.dump('unlabeled_infer_cfg.py')
print('Generated unlabeled_infer_cfg.py')
