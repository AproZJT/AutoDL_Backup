_base_ = './grounding_dino_swin-b_inference_zerowaste_f.py'

# 核心魔法：注入负面词汇吸收背景噪点
# 注意：前 4 个类别的顺序绝对不能变，保证评测 ID 对齐！
model = dict(
    test_cfg=dict(
        chunked_size=40,
    )
)

# 覆盖数据集的 text_prompt
test_dataloader = dict(
    dataset=dict(
        text_prompt="rigid plastic . cardboard . metal . soft plastic . floor . shadow . reflection . unknown garbage ."
    )
)
