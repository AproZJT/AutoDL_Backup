from mmengine.config import Config
import sys

probe_ckpt = sys.argv[1]
# 直接继承我们刚才写好的稳健版探测配置
cfg = Config.fromfile('external_modules/mmdetection/configs/grounding_dino/grounding_dino_swin-b_small_object_probe.py')

# 💉 替换起点：站在 0.541 的肩膀上继续微调检测头
cfg.load_from = probe_ckpt
# 继续跑 1 圈（事实上的第 2 圈）
cfg.train_cfg.max_epochs = 1

cfg.dump('external_modules/mmdetection/configs/grounding_dino/grounding_dino_swin-b_small_object_round2.py')
print("✅ 第2圈冲刺配置生成成功！")
