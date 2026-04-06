_base_ = './grounding_dino_swin-b_semi-sup_zerowaste.py'

# 专门针对无标签 Test 测试集重写数据流管道
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # 💥 这里极其干净，绝对没有 LoadAnnotations！
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities'))
]

# 强制将纯净管道应用到 test_dataloader
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
