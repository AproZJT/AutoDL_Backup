import argparse
import torch
from collections import OrderedDict

def main():
    parser = argparse.ArgumentParser(description="Offline EMA Checkpoint Averaging")
    parser.add_argument('--inputs', nargs='+', required=True, help='List of checkpoint paths')
    parser.add_argument('--weights', nargs='+', type=float, required=True, help='Weights for each checkpoint')
    parser.add_argument('--output', type=str, required=True, help='Output path for the fused checkpoint')
    args = parser.parse_args()

    assert len(args.inputs) == len(args.weights), "输入权重的数量必须与权值数量一致！"

    # 归一化权重 (例如传入 0.7 0.3，相加为 1.0)
    sum_w = sum(args.weights)
    weights = [w / sum_w for w in args.weights]

    print(f"\n[INFO] 正在执行离线 EMA 融合...")
    for path, w in zip(args.inputs, weights):
        print(f"       -> {path} (Weight: {w:.2f})")

    # 全部加载到 CPU 防止显存爆炸
    ckpts = [torch.load(p, map_location='cpu') for p in args.inputs]

    # 以第一个权重 (Teacher) 为基底，保留其完整的 meta 结构
    base_ckpt = ckpts[0]
    base_state_dict = base_ckpt.get('state_dict', base_ckpt)
    
    averaged_state_dict = OrderedDict()

    # 遍历所有参数名称
    for key in base_state_dict.keys():
        # 如果是浮点数张量，执行加权平均
        if isinstance(base_state_dict[key], torch.Tensor) and base_state_dict[key].is_floating_point():
            averaged_state_dict[key] = torch.zeros_like(base_state_dict[key])
            for i, ckpt in enumerate(ckpts):
                state_dict = ckpt.get('state_dict', ckpt)
                if key in state_dict:
                    averaged_state_dict[key] += state_dict[key] * weights[i]
        else:
            # 如果是整数 (如统计 step/epoch) 或非张量，直接继承 Teacher 的值
            averaged_state_dict[key] = base_state_dict[key]

    # 将融合后的参数写回
    if 'state_dict' in base_ckpt:
        base_ckpt['state_dict'] = averaged_state_dict
    else:
        base_ckpt = averaged_state_dict

    torch.save(base_ckpt, args.output)
    print(f"[SUCCESS] EMA 融合完成！已保存至: {args.output}\n")

if __name__ == '__main__':
    main()
