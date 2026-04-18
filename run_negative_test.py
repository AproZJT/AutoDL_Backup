import os
from mmengine.config import Config

# 加载你的 SOTA 配置
cfg = Config.fromfile('external_modules/mmdetection/configs/grounding_dino/grounding_dino_swin-b_inference_zerowaste_f.py')

# 偷天换日：将模型替换为我们的拦截器
cfg.model.type = 'GroundingDINONegativeAbsorber'

# 修改 Pipeline：插入输入端文本注入节点
pipeline = cfg.test_dataloader.dataset.pipeline
for i, trans in enumerate(pipeline):
    if trans['type'] == 'PackDetInputs':
        pipeline.insert(i, dict(type='InjectNegativePrompt'))
        # 确保打包器能够接收文本变量
        meta_keys = list(trans.get('meta_keys', []))
        if 'text' not in meta_keys:
            meta_keys.extend(['text', 'custom_entities'])
            trans['meta_keys'] = tuple(meta_keys)
        break
cfg.test_dataloader.dataset.pipeline = pipeline

# 注册我们的两个外挂模块
cfg.custom_imports = dict(imports=['negative_prompt_transform', 'filter_negative_predictions'], allow_failed_imports=False)

# 保存最终探针配置并执行
cfg.dump('test_negative_prompt_cfg.py')
print("✅ 负面提示词黑洞架构组装完成，正在点火验证...")
os.system('python external_modules/mmdetection/tools/test.py test_negative_prompt_cfg.py weights/final_sota/best_swa_0.545.pth')
