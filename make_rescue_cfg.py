from mmengine.config import Config
cfg = Config.fromfile('rigorous_finetune_cfg_v3.py')

# 核心抢救操作：彻底关闭多进程数据加载，释放所有 CPU RAM
cfg.train_dataloader.num_workers = 0
cfg.train_dataloader.persistent_workers = False

# 强行把单卡 Batch Size 压到 1，防止显存爆炸
cfg.train_dataloader.batch_size = 1

cfg.dump('rescue_finetune_cfg.py')
print("✅ 抢救版超低内存 Config 已生成！")
