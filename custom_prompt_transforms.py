from mmdet.registry import TRANSFORMS
from mmcv.transforms import BaseTransform

@TRANSFORMS.register_module()
class InjectPrompt2(BaseTransform):
    def transform(self, results):
        results['text'] = 'hard plastic . cardboard box . metal scrap . soft plastic bag .'
        results['custom_entities'] = True
        return results

@TRANSFORMS.register_module()
class InjectPrompt3(BaseTransform):
    def transform(self, results):
        results['text'] = 'plastic container . corrugated cardboard . metallic waste . film plastic .'
        results['custom_entities'] = True
        return results
