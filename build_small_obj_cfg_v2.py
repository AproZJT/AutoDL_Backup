from mmengine.config import Config
import sys

best_ckpt = sys.argv[1]
cfg = Config.fromfile('external_modules/mmdetection/configs/grounding_dino/grounding_dino_swin-b_semi-sup_zerowaste.py')

# 🌟 听你的：保守尺度 + 拒绝空裁剪！
small_obj_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            # 路线 A：稳住基本盘
            [dict(type='RandomChoiceResize', scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333), (608, 1333), (640, 1333), (672, 1333), (704, 1333), (736, 1333), (768, 1333), (800, 1333)], keep_ratio=True)],
            # 路线 B：极其稳健的小目标专项放大+裁剪
            [
                # ⚠️ 修改 1：砍掉激进大尺度，防 OOM
                dict(type='RandomChoiceResize', scales=[(800, 2000), (960, 2400)], keep_ratio=True),
                # ⚠️ 修改 2：关闭 allow_negative_crop，只裁有目标的区域！
                dict(type='RandomCrop', crop_type='absolute_range', crop_size=(600, 800), allow_negative_crop=False),
                dict(type='RandomChoiceResize', scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333), (608, 1333), (640, 1333), (672, 1333), (704, 1333), (736, 1333), (768, 1333), (800, 1333)], keep_ratio=True)
            ]
        ]
    ),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction', 'text', 'custom_entities'))
]

def replace_pipeline(d):
    if isinstance(d, dict):
        if 'pipeline' in d:
            d['pipeline'] = small_obj_pipeline
        for k, v in d.items():
            replace_pipeline(v)
    elif isinstance(d, list):
        for item in d:
            replace_pipeline(item)

replace_pipeline(cfg.train_dataloader)

# 💉 注入 1 Epoch 探测策略和 Head-Only 冻结
cfg.load_from = best_ckpt
cfg.optim_wrapper.optimizer.lr = 1e-6
cfg.optim_wrapper.paramwise_cfg = dict(custom_keys={'backbone': dict(lr_mult=0.0, decay_mult=0.0)})
# ⚠️ 修改 3：只跑 1 圈测风向！
cfg.train_cfg.max_epochs = 1
cfg.train_cfg.val_interval = 1
cfg.param_scheduler = [dict(type='ConstantLR', factor=1.0, by_epoch=True, begin=0, end=1)]

cfg.dump('external_modules/mmdetection/configs/grounding_dino/grounding_dino_swin-b_small_object_probe.py')
print("✅ 稳健探测版配置生成成功！")
