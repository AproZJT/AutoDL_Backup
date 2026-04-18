_base_ = ['grounding_dino_swin-b_finetune_zerowaste_f.py']

# ==============================================================================
# 1. 继承最优 Teacher 权重，压低学习率
# ==============================================================================
load_from = 'weights/final_sota/best_swa_0.545.pth'

optim_wrapper = dict(
    optimizer=dict(lr=2e-5) # 仅覆盖 lr，保留基座的 weight_decay 和 paramwise_cfg
)

# ==============================================================================
# 2. 显式定义 Grounding DINO 专属 Pipeline (关键修复：确保 text 字段不丢)
# ==============================================================================
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    # 核心修复点：PackDetInputs 必须打包 text 和 custom_entities
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities'))
]

# ==============================================================================
# 3. 数据集与混合策略 (真 2 : 伪 1)
# ==============================================================================
class_name = ('rigid_plastic', 'cardboard', 'metal', 'soft_plastic')
dataset_type = 'CocoDataset'

labeled_dataset = dict(
    type=dataset_type,
    data_root='data/zerowaste-f/',
    ann_file='train/labels.json',
    data_prefix=dict(img='train/data/'),
    return_classes=True,  # 核心修复点：强制要求 Dataset 返回类别文本
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    pipeline=train_pipeline,
    metainfo=dict(classes=class_name)
)

pseudo_dataset = dict(
    type=dataset_type,
    data_root='data/zerowaste-f/',
    ann_file='final_pseudo_coco.json',
    data_prefix=dict(img='train/data/'),
    return_classes=True,  # 核心修复点：强制要求 Dataset 返回类别文本
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    pipeline=train_pipeline,
    metainfo=dict(classes=class_name)
)

# 使用 _delete_=True 干净利落地覆盖基座的 dataset 结构
train_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type='ConcatDataset',
        datasets=[labeled_dataset, labeled_dataset, pseudo_dataset]
    )
)
