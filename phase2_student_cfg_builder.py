from mmengine.config import Config
from copy import deepcopy
import os

BASE_SWINT_CFG = 'external_modules/mmdetection/configs/grounding_dino/grounding_dino_swin-t_finetune_16xb2_1x_coco.py'
cfg = Config.fromfile(BASE_SWINT_CFG)

# --- 1. 强行注入 ZeroWaste-f 专属类别和属性 ---
classes = ('rigid_plastic', 'cardboard', 'metal', 'soft_plastic')
metainfo = dict(classes=classes)

def set_ds_info(ds, ann, img_prefix):
    """递归深入修改数据集配置的杀手锏函数"""
    if 'dataset' in ds:
        set_ds_info(ds['dataset'], ann, img_prefix)
        return
    ds['type'] = 'CocoDataset'
    ds['data_root'] = './data/'
    ds['ann_file'] = ann
    ds['data_prefix'] = dict(img=img_prefix)
    ds['metainfo'] = metainfo
    # 清理掉官方 COCO 自带的过严过滤规则
    if 'filter_cfg' in ds:
        ds['filter_cfg'] = dict(filter_empty_gt=False, min_size=0)

# --- 2. 修正官方训练集路径 ---
set_ds_info(cfg.train_dataloader.dataset, 'zerowaste-f/train/labels.json', 'zerowaste-f/train/data/')
original_train_ds = deepcopy(cfg.train_dataloader.dataset)

# --- 3. 构建伪标签数据集 ---
pseudo_dataset = deepcopy(cfg.train_dataloader.dataset)
# 自动探测伪标签对应的图片文件夹
pseudo_img_prefix = 'zerowaste-f/unlabeled/data/'
if not os.path.exists('./data/' + pseudo_img_prefix):
    pseudo_img_prefix = 'zerowaste-f/train/data/'
set_ds_info(pseudo_dataset, 'zerowaste-f/teacher_pseudo_labels.json', pseudo_img_prefix)

# --- 4. 组合大杀器：ConcatDataset ---
cfg.train_dataloader.dataset = dict(
    type='ConcatDataset',
    datasets=[original_train_ds, pseudo_dataset]
)

# --- 5. 修正 Val 和 Test 路径 (防止评测时找错账本) ---
if hasattr(cfg, 'val_dataloader'):
    set_ds_info(cfg.val_dataloader.dataset, 'zerowaste-f/val/labels.json', 'zerowaste-f/val/data/')
    cfg.val_evaluator['ann_file'] = './data/zerowaste-f/val/labels.json'
if hasattr(cfg, 'test_dataloader'):
    set_ds_info(cfg.test_dataloader.dataset, 'zerowaste-f/test/labels.json', 'zerowaste-f/test/data/')
    cfg.test_evaluator['ann_file'] = './data/zerowaste-f/test/labels.json'

# --- 6. 小卡防闪退与梯度累加 (保留之前的精华) ---
cfg.train_dataloader.num_workers = 0
cfg.train_dataloader.persistent_workers = False
cfg.train_dataloader.batch_size = 2

if 'optim_wrapper' not in cfg:
    cfg.optim_wrapper = dict()
cfg.optim_wrapper.accumulative_counts = 8

cfg.train_cfg.max_epochs = 12
cfg.train_cfg.val_interval = 1

cfg.work_dir = './work_dirs/phase2_swinT_student'
cfg.dump('phase2_student_finetune.py')
print("✅ Student 专属防 OOM 配置文件已生成 (已完美切入 ZeroWaste-f 赛道): phase2_student_finetune.py")
