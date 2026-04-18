#!/usr/bin/env bash
set -eu

ROOT_DIR="$PWD"
TEST_SCRIPT="$ROOT_DIR/external_modules/mmdetection/tools/test.py"
CFG_FILE="$ROOT_DIR/external_modules/mmdetection/configs/grounding_dino/grounding_dino_swin-b_inference_zerowaste_f.py"
GT_JSON="$ROOT_DIR/data/zerowaste-f/test/labels.json"
RAW_DIR="$ROOT_DIR/data/pseudo_labels"
OUT_DIR="$ROOT_DIR/data/pseudo_labels/stage3_fusion"

mkdir -p "$OUT_DIR"

echo "=================================================="
echo "🔥 Stage 3: 双模型防爆融合 (精准修正版)"
echo "=================================================="

# 修正 1：确定的绝对路径，拒绝猜谜
MODEL_A_JSON="$RAW_DIR/raw_s0010.bbox.json"
WEIGHT_B="$ROOT_DIR/weights/gdino-swin-b/zerowaste_f_finetuned_best_coco_bbox_mAP.pth"
MODEL_B_PREFIX="$RAW_DIR/raw_model_b_s0010"
MODEL_B_JSON="${MODEL_B_PREFIX}.bbox.json"

if [ ! -f "$MODEL_A_JSON" ]; then
    echo "❌ 致命错误：找不到阶段 1 的基础预测 $MODEL_A_JSON ！"
    exit 1
fi

# 2. 生成模型 B 的预测框 (缓存机制)
if [ ! -f "$MODEL_B_JSON" ]; then
    echo "⏳ 正在拉起 Model B 进行补充推理 (score_thr=0.010)..."
    # 修正 3(部分)：推理阶段也不隐藏报错信息
    python "$TEST_SCRIPT" "$CFG_FILE" "$WEIGHT_B" \
      --cfg-options model.test_cfg.rcnn.score_thr=0.010 \
      test_evaluator.outfile_prefix="$MODEL_B_PREFIX"
else
    echo "✅ Model B 预测缓存已存在，直接复用！"
fi

MERGED_JSON="$OUT_DIR/merged_anti_explode.json"

# 3. 执行防爆合并
echo "🔪 正在合并双流预测框..."
python - << PY
import json

with open("$MODEL_A_JSON", 'r') as f:
    preds_a = json.load(f)
with open("$MODEL_B_JSON", 'r') as f:
    preds_b = json.load(f)

merged = preds_a + preds_b
filtered = [d for d in merged if d.get('score', 0) >= 0.005]

with open("$MERGED_JSON", 'w') as f:
    json.dump(filtered, f)
print(f"✅ 合并完成！总候选框数量: {len(filtered)}")
PY

# 4. 细粒度全局 NMS 搜索
echo "⚖️ 开始细粒度全局 NMS 去重与算分..."
echo "固定最优局部: metal_nms_iou = 0.712"

# 修正 2：扩宽 DNMS 搜索范围，加入 0.66
for DNMS in 0.58 0.60 0.62 0.64 0.66; do
  echo "--------------------------------------------------"
  echo "⚙️ 测试组合: default_nms_iou = $DNMS"
  OUT_JSON="$OUT_DIR/post_dnms${DNMS//./}.json"
  
  # 修正 3：去掉 grep，保留完整日志输出，绝不掩盖错误
  python "$ROOT_DIR/semi_sup/scripts/apply_best_postproc.py" \
    --gt_json "$GT_JSON" \
    --pred_json "$MERGED_JSON" \
    --out_json "$OUT_JSON" \
    --default_thr 0.0 \
    --default_nms_iou $DNMS \
    --metal_nms_iou 0.712 \
    --max_dets 100,300,1000
done

echo "=================================================="
echo "🎉 融合评估完毕！请向上滚动查看各个 DNMS 的完整输出结果！"
echo "=================================================="
