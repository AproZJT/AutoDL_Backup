from mmengine.config import Config

cfg = Config.fromfile('external_modules/mmdetection/configs/grounding_dino/grounding_dino_swin-b_inference_zerowaste_f.py')
cfg.test_dataloader.dataset.ann_file = 'zerowaste-f/test/labels.json'
cfg.test_dataloader.dataset.data_prefix = dict(img='test/data/')
cfg.test_evaluator.ann_file = './data/zerowaste-f/test/labels.json'
cfg.test_evaluator.outfile_prefix = './data/pseudo_labels/final_sota_tta_raw'
cfg.model.test_cfg.rcnn.score_thr = 0.0
cfg.dump('tta_test_cfg_tmp_resolved.py')
print('generated tta_test_cfg_tmp_resolved.py')
