_base_ = './grounding_dino_swin-b_semi-sup_zerowaste.py'
tta_model = dict(
    type='DetTTAModel',
    tta_cfg=dict(
        nms=dict(type='nms', iou_threshold=0.55),
        max_per_img=100))
img_scales = [(1333, 800), (1333, 960), (1333, 1120)] # 👈 核心改动：加入超大尺度
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='TestTimeAug',
        transforms=[
            [dict(type='Resize', scale=s, keep_ratio=True) for s in img_scales],
            [dict(type='RandomFlip', prob=1.), dict(type='RandomFlip', prob=0.)],
            [dict(type='LoadAnnotations', with_bbox=True)],
            [dict(type='PackDetInputs',
                  meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                             'scale_factor', 'flip', 'flip_direction',
                             'text', 'custom_entities'))]
        ])
]
