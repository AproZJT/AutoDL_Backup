_base_ = './grounding_dino_swin-b_semi-sup_zerowaste.py'

load_from = 'experiments/swin-b_HeadOnly_Finetune_FINAL/best_coco_bbox_mAP_epoch_2.pth'

# 🌟 绝杀核心：重写数据流管道，加入大尺度放大与随机裁剪！
# 注意：Grounding DINO 强依赖文本 prompt，必须带上 text 字段
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            # 路线 A：常规多尺度训练（稳住基本盘）
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            # 路线 B：小目标杀手锏！（暴力放大 + 随机裁剪）
            [
                # 1. 先把图片极其暴力地放大 (强迫小目标变大)
                dict(
                    type='RandomChoiceResize',
                    scales=[(800, 2000), (1000, 2500), (1200, 3000)],
                    keep_ratio=True),
                # 2. 随机裁出一块正常大小的区域喂给模型
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(600, 800),
                    allow_negative_crop=True),
                # 3. 最后规范化尺寸
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333),
                            (576, 1333), (608, 1333), (640, 1333),
                            (672, 1333), (704, 1333), (736, 1333),
                            (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]
    ),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction',
                   'text', 'custom_entities'))
]

# 挂载最新的 3:1 真伪黄金数据集和增强管道
train_dataloader = dict(
    dataset=dict(
        ann_file='data/pseudo_labels/zerowaste_3to1_mixed_finetune.json',
        pipeline=train_pipeline
    )
)

# 继续保持防遗忘的 Head-Only 极低学习率微调
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-6, weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.0, decay_mult=0.0)}
    )
)

# 也是只打两枪，见好就收！
train_cfg = dict(max_epochs=2, val_interval=1)
