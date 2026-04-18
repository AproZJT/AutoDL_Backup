from mmdet.registry import TRANSFORMS
from mmcv.transforms import BaseTransform

@TRANSFORMS.register_module()
class InjectNegativePrompt(BaseTransform):
    def transform(self, results):
        # 前4个是真实类别，后4个是用来吸纳噪点的"黑洞类别"
        results['text'] = 'rigid plastic . cardboard . metal . soft plastic . floor . shadow . reflection . unknown background .'
        results['custom_entities'] = True
        return results
