#!/bin/bash
set -e
echo "[INFO] 开始执行 6 组参数的自动化短训 (Mini-Training)..."

export HF_ENDPOINT=https://hf-mirror.com

cd external_modules/mmdetection

CONFIG="configs/grounding_dino/grounding_dino_swin-b_semi-sup_zerowaste.py"
BASE_JSON_DIR="../../data/pseudo_labels/sweeps"
TARGET_JSON="../../data/pseudo_labels/zerowaste-s_consolidated_pseudo_annotations.json"

declare -a jsons=(
  "run_tauF035/phase1_ensemble_tauF0p35_theta0p70_votes2_snmsIou0p55.json"
  "run_tauF045/phase1_ensemble_tauF0p45_theta0p70_votes2_snmsIou0p55.json"
  "run_tauF035/phase1_ensemble_tauF0p35_theta0p65_votes2_snmsIou0p55.json"
  "run_tauF040/phase1_ensemble_tauF0p40_theta0p65_votes2_snmsIou0p55.json"
  "run_tauF030/phase1_ensemble_tauF0p30_theta0p60_votes2_snmsIou0p45.json"
  "run_tauF040/phase1_ensemble_tauF0p40_theta0p70_votes2_snmsIou0p55.json"
)

for json_file in "${jsons[@]}"; do
  echo "========================================================="
  echo "[INFO] 正在部署伪标签: $json_file"
  echo "========================================================="
  
  cp "$BASE_JSON_DIR/$json_file" "$TARGET_JSON"

  filename=$(basename -- "$json_file")
  run_name="${filename%.*}"
  WORK_DIR="work_dirs/mini_train_$run_name"

  python tools/train.py $CONFIG     --work-dir "$WORK_DIR"     --cfg-options train_cfg.max_epochs=2 default_hooks.checkpoint.interval=2
    
  echo "[INFO] $run_name 短训完成！"
done
