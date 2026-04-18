from mmengine.config import Config

cfg = Config.fromfile('external_modules/mmdetection/configs/grounding_dino/grounding_dino_swin-b_inference_zerowaste_f.py')

# 对齐到 zerowaste-f/test
cfg.test_dataloader.dataset.ann_file = 'test/labels.json'
cfg.test_dataloader.dataset.data_prefix = dict(img='test/data/')
cfg.test_dataloader.num_workers = 0
cfg.test_dataloader.persistent_workers = False
cfg.test_evaluator.ann_file = './data/zerowaste-f/test/labels.json'
cfg.test_evaluator.outfile_prefix = './data/pseudo_labels/final_sota_tta_raw'
cfg.model.test_cfg.rcnn.score_thr = 0.0

# 关键：自定义 tta_pipeline，PackDetInputs 必须包含 text/custom_entities
cfg.tta_model = dict(type='DetTTAModel', tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.5), max_per_img=1000))
cfg.tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [dict(type='RandomFlip', prob=1.0), dict(type='RandomFlip', prob=0.0)],
            [dict(type='FixScaleResize', scale=(800, 1333), keep_ratio=True)],
            [dict(
                type='PackDetInputs',
                meta_keys=(
                    'img_id','img_path','ori_shape','img_shape','scale_factor',
                    'flip','flip_direction','text','custom_entities'
                )
            )]
        ])
]

cfg.dump('gdino_tta_test_cfg.py')
print('generated gdino_tta_test_cfg.py')
