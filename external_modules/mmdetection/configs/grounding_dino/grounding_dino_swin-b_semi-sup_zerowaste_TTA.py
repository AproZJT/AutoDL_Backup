_base_ = './grounding_dino_swin-b_semi-sup_zerowaste.py'

# 开启 TTA 融合机制，设置 NMS 阈值把 4 个视角的框合并
tta_model = dict(
    type='DetTTAModel',
    tta_cfg=dict(
        nms=dict(type='nms', iou_threshold=0.55),
        max_per_img=100))

# 轻量多尺度：原图标准尺寸 + 放大版（专抓小目标）
img_scales = [(1333, 800), (1333, 960)]

# 定义 TTA 数据流管道
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [dict(type='Resize', scale=s, keep_ratio=True) for s in img_scales],
            [
                dict(type='RandomFlip', prob=1.),
                dict(type='RandomFlip', prob=0.)
            ],
            [dict(type='LoadAnnotations', with_bbox=True)],
            [
                dict(
                    type='PackDetInputs',
                    # Grounding DINO 核心：必须保留 text 和 custom_entities 否则会报错！
                    meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                               'scale_factor', 'flip', 'flip_direction',
                               'text', 'custom_entities'))
            ]
        ])
]
