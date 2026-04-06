_base_ = './grounding_dino_swin-b_semi-sup_zerowaste.py'

# 站在巨人的肩膀上
load_from = 'experiments/swin-b_semi-sup_Round2_FINAL/best_coco_bbox_mAP_epoch_11.pth'

# 🌟 绝杀核心：锁死骨干，极低学习率！
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-6, weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys={
            # 强行给 backbone 踩刹车，彻底冻结防止灾难性遗忘！
            'backbone': dict(lr_mult=0.0, decay_mult=0.0)
        }
    )
)

# 极短周期：只跑 2 圈，随时盯盘
train_cfg = dict(max_epochs=2, val_interval=1)

# 废弃复杂的衰减策略，保持 1e-6 平稳走完
param_scheduler = [
    dict(type='ConstantLR', factor=1.0, by_epoch=True, begin=0, end=2)
]
