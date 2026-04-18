_base_ = ['grounding_dino_swin-b_finetune_zerowaste_f.py']

# High-start initialization from the best Teacher checkpoint
load_from = 'weights/final_sota/best_swa_0.545.pth'

# Keep a conservative fine-tuning learning rate
optim_wrapper = dict(
    optimizer=dict(lr=2e-5),
)

# Reuse the Grounding DINO-compatible pipeline from the base config
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
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities'))
]

class_name = ('rigid_plastic', 'cardboard', 'metal', 'soft_plastic')
dataset_type = 'CocoDataset'

labeled_dataset = dict(
    type=dataset_type,
    data_root='data/zerowaste-f/',
    ann_file='train/labels.json',
    data_prefix=dict(img='train/data/'),
    return_classes=True,
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    pipeline=train_pipeline,
    metainfo=dict(classes=class_name),
)

pseudo_dataset = dict(
    type=dataset_type,
    data_root='data/zerowaste-f/',
    ann_file='final_pseudo_coco.json',
    data_prefix=dict(img='train/data/'),
    return_classes=True,
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    pipeline=train_pipeline,
    metainfo=dict(classes=class_name),
)

# 3:1 mix — favor real labels to keep training stable
train_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type='ConcatDataset',
        datasets=[labeled_dataset, labeled_dataset, labeled_dataset, pseudo_dataset]
    )
)
