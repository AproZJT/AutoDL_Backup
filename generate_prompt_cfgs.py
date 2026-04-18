from mmengine.config import Config

def inject_transform(cfg_path, transform_name, out_path):
    cfg = Config.fromfile(cfg_path)
    cfg.custom_imports = dict(imports=['custom_prompt_transforms'], allow_failed_imports=False)

    pipeline = cfg.test_dataloader.dataset.pipeline
    for i, trans in enumerate(pipeline):
        if trans['type'] == 'PackDetInputs':
            pipeline.insert(i, dict(type=transform_name))
            meta_keys = list(trans.get('meta_keys', []))
            for k in ['text', 'custom_entities']:
                if k not in meta_keys:
                    meta_keys.append(k)
            trans['meta_keys'] = tuple(meta_keys)
            break

    cfg.test_dataloader.dataset.pipeline = pipeline
    cfg.dump(out_path)

base = 'external_modules/mmdetection/configs/grounding_dino/grounding_dino_swin-b_inference_zerowaste_f.py'
inject_transform(base, 'InjectPrompt2', 'prompt2_cfg.py')
inject_transform(base, 'InjectPrompt3', 'prompt3_cfg.py')
print('OK: prompt2_cfg.py, prompt3_cfg.py generated')
