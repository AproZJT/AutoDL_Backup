from mmdet.registry import MODELS
from mmdet.models.detectors.grounding_dino import GroundingDINO

@MODELS.register_module()
class GroundingDINONegativeAbsorber(GroundingDINO):
    def predict(self, batch_inputs, batch_data_samples, rescale=True):
        # 让原始模型用 8 个类别正常推理
        results = super().predict(batch_inputs, batch_data_samples, rescale)
        
        # 在送交评测前，强行删掉掉进后 4 个黑洞类别的框
        for data_sample in results:
            preds = data_sample.pred_instances
            # 只保留前 4 个合法类别的预测框
            valid_mask = preds.labels < 4
            data_sample.pred_instances = preds[valid_mask]
        return results
