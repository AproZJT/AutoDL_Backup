_base_ = './grounding_dino_swin-b_semi-sup_zerowaste.py'

# 🌟 核心 1：站在巨人的肩膀上！从 0.543 权重开始
load_from = 'experiments/swin-b_semi-sup_Round2_FINAL/best_coco_bbox_mAP_epoch_11.pth'

# 🌟 核心 2：极其温柔的学习率 (从原来的 5e-5 骤降到 5e-6)
# 防止破坏原本极其优秀的特征表示
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=5e-6, weight_decay=0.0001))

# 🌟 核心 3：极其克制的训练周期 (只跑 3 圈，每圈查分早停！)
train_cfg = dict(max_epochs=3, val_interval=1)

# 取消长周期的学习率衰减，保持平稳
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=100),
    dict(type='MultiStepLR', begin=0, end=3, by_epoch=True, milestones=[2], gamma=0.5)
]
