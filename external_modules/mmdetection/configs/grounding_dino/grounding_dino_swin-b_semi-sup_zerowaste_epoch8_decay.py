_base_ = './grounding_dino_swin-b_semi-sup_zerowaste.py'

# 核心魔法：覆盖学习率调度器，第 8 圈将学习率降到十分之一 (5e-6)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8],  # 👈 就在这里！第 8 圈精准降速！
        gamma=0.1)
]
