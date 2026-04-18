_base_ = './grounding_dino_swin-b_inference_zerowaste_f.py'

tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale=(1333, 800), keep_ratio=True),
                dict(type='Resize', scale=(1666, 1000), keep_ratio=True),
            ],
            [
                dict(type='RandomFlip', prob=0.0),
                dict(type='RandomFlip', prob=1.0),
            ],
            [
                dict(
                    type='PackDetInputs',
                    meta_keys=(
                        'img_id', 'img_path', 'ori_shape', 'img_shape',
                        'scale_factor', 'flip', 'flip_direction',
                        'text', 'custom_entities'
                    ))
            ]
        ])
]

# 最后一搏：将 iou_threshold 下调至 0.5
tta_model = dict(
    type='DetTTAModel',
    tta_cfg=dict(
        nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.001),
        max_per_img=300))
