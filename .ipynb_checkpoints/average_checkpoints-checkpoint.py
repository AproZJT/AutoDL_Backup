import torch
import os

def blend_checkpoints(ckpt_paths, output_path):
    print(f"🚀 准备融合 {len(ckpt_paths)} 个模型的权重...")
    
    # 验证文件是否存在
    for p in ckpt_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"找不到权重文件: {p}")

    # 1. 以第一个模型为基底加载
    print(f"正在加载基底模型: {ckpt_paths[0]}")
    base_ckpt = torch.load(ckpt_paths[0], map_location='cpu')
    base_state = base_ckpt['state_dict']
    
    # 初始化累加字典
    averaged_state = {k: v.clone() for k, v in base_state.items()}
    
    # 2. 累加后续所有模型的权重
    for path in ckpt_paths[1:]:
        print(f"正在加载并累加模型: {path}")
        ckpt = torch.load(path, map_location='cpu')
        state = ckpt['state_dict']
        for k in averaged_state.keys():
            averaged_state[k] += state[k]
            
    # 3. 求算术平均
    num_ckpts = len(ckpt_paths)
    print(f"正在计算 {num_ckpts} 个权重的算术平均值...")
    for k in averaged_state.keys():
        if averaged_state[k].is_floating_point():
            averaged_state[k].div_(num_ckpts)
        else:
            # 针对非浮点数（例如跟踪的 batch 数量）使用整除
            averaged_state[k] //= num_ckpts
            
    # 4. 替换基底模型的权重并保存
    base_ckpt['state_dict'] = averaged_state
    torch.save(base_ckpt, output_path)
    print(f"\n✅ 权重融合大功告成！融合后的 SOTA 模型已保存至: {output_path}")

if __name__ == "__main__":
    # ================= 配置你想融合的仙丹 =================
    # 我们把原作者的最强基线，加上你刚才跑出来的策略A的精华，全部融合！
    checkpoints_to_blend = [
        # 1. 原始满分权重 (0.543)
        "weights/gdino-swin-b/zerowaste_semi-sup_best_coco_bbox_mAP.pth",
        
        # 2. 刚才策略A微调出来的新 SOTA (0.544)
        "experiments/swin-b_strategy_A_digest/best_coco_bbox_mAP_epoch_2.pth",
        
        # （如果你在 strategy_A 文件夹里还有 epoch_3.pth，也可以加进来）
        # "experiments/swin-b_strategy_A_digest/epoch_3.pth"
    ]
    
    out_file = "weights/gdino-swin-b/zerowaste_SWA_fused_ultimate.pth"
    
    blend_checkpoints(checkpoints_to_blend, out_file)