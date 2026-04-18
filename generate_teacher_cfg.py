from mmengine.config import Config
cfg = Config.fromfile('external_modules/mmdetection/configs/grounding_dino/grounding_dino_swin-b_inference_zerowaste_f.py')

# 1. 替换数据输入端
cfg.test_dataloader.dataset.ann_file = 'unlabeled/labels.json'
cfg.test_dataloader.dataset.data_prefix = dict(img='train/data/')

# 2. 替换验证对账端 (防止 ID 匹配 AssertionError)
unlabeled_abs_path = './data/zerowaste-f/unlabeled/labels.json'
if isinstance(cfg.test_evaluator, dict):
    cfg.test_evaluator.ann_file = unlabeled_abs_path
elif isinstance(cfg.test_evaluator, list):
    for metric in cfg.test_evaluator:
        if 'ann_file' in metric:
            metric.ann_file = unlabeled_abs_path

cfg.dump('teacher_infer_cfg.py')
