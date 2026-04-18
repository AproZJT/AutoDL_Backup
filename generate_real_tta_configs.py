import os
from mmengine.config import Config

base_cfg_path = 'external_modules/mmdetection/configs/grounding_dino/grounding_dino_swin-b_inference_zerowaste_f.py'

# 生成 0.9x 尺度 (缩小)
cfg_90 = Config.fromfile(base_cfg_path)
pipeline_90 = cfg_90.test_dataloader.dataset.pipeline
for transform in pipeline_90:
    if transform['type'] == 'FixScaleResize':
        transform['scale'] = (1200, 720) # 约 0.9 倍
cfg_90.test_dataloader.dataset.pipeline = pipeline_90
cfg_90.dump('tta_s90_cfg.py')

# 生成 1.1x 尺度 (放大)
cfg_110 = Config.fromfile(base_cfg_path)
pipeline_110 = cfg_110.test_dataloader.dataset.pipeline
for transform in pipeline_110:
    if transform['type'] == 'FixScaleResize':
        transform['scale'] = (1466, 880) # 约 1.1 倍
cfg_110.test_dataloader.dataset.pipeline = pipeline_110
cfg_110.dump('tta_s110_cfg.py')

print("✅ 0.9x 和 1.1x 多尺度配置文件已就绪！")
