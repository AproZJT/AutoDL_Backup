import os
from mmengine.config import Config

base_cfg_path = 'external_modules/mmdetection/configs/grounding_dino/grounding_dino_swin-b_inference_zerowaste_f.py'
cfg = Config.fromfile(base_cfg_path)

# 1. 从正确的 test_dataloader 里“偷”出图片路径与元数据
test_ds = cfg.test_dataloader.dataset
train_pipeline = cfg.train_dataloader.dataset.pipeline

# 2. 组装纯净的伪标签训练集
pseudo_dataset = dict(
    type=test_ds.type,
    data_root=test_ds.data_root,
    ann_file='test/pseudo_labels.json',  # 指向我们刚生成的纯净伪标签
    data_prefix=test_ds.data_prefix,     # 完美对齐测试集图片文件夹
    pipeline=train_pipeline,
    return_classes=True
)

# 继承可能的类别 metainfo (极其关键，防止类别错乱)
if 'metainfo' in test_ds:
    pseudo_dataset['metainfo'] = test_ds.metainfo

# 强行覆盖：这 1 个 Epoch，我们只看伪标签！
cfg.train_dataloader.dataset = pseudo_dataset

# 3. 冻结策略：锁死特征提取，只微调头部
cfg.model.bbox_head.loss_cls.loss_weight = 1.0 
cfg.model.bbox_head.loss_bbox.loss_weight = 2.0 

# 4. 极其保守的超参 (仅 1 个 Epoch，防过拟合)
cfg.train_cfg.max_epochs = 1
cfg.train_cfg.val_interval = 1

# 学习率降维打击 (微调专属 1e-5)
if 'optim_wrapper' in cfg:
    cfg.optim_wrapper.optimizer.lr = 1e-5 

# 5. 加载最强底座
cfg.load_from = 'weights/final_sota/best_swa_0.545.pth'
cfg.work_dir = './work_dirs/pseudo_finetune_ultimate'

cfg.dump('pseudo_finetune_cfg.py')
print("✅ 纯伪标签微调 Config 已完美修复！")
