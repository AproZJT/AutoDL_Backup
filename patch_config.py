import os

file_path = 'external_modules/mmdetection/configs/grounding_dino/grounding_dino_swin-b_semi-sup_zerowaste_finetune.py'

with open(file_path, 'r') as f:
    content = f.read()

insert_str = """
    # ====== 👇 新增的 Exp-C 强增强算子 👇 ======
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    # ==============================================
"""

# 检查是否已经插入过了，防止重复插入
if 'PhotoMetricDistortion' not in content:
    # 精准替换注入
    new_content = content.replace(
        "dict(type='RandomFlip', prob=0.5),", 
        f"dict(type='RandomFlip', prob=0.5),{insert_str}"
    )
    with open(file_path, 'w') as f:
        f.write(new_content)
    print("✅ [SUCCESS] 强增强算子 PhotoMetricDistortion 已成功安全注入配置文件！")
else:
    print("⚠️ [INFO] 配置文件中已存在该算子，无需重复插入。")
